import time
import sys
import os
sys.path += ['../']
import tensorflow as tf
from tensorflow import keras
import logging
import datetime
from utils.mask_utils import *
import numpy as np
from pprint import pprint
from utils.validation_metrics import get_D_classifier_report, get_Discriminator_where_report


def calculate_rewards(d_classify_output,d_where_output, 
        fake_mask_label,max_len=None,
        modified_expected_label=1):
    """
    input:
    d_classify_output [bs,1]
    d_where_output [bs,seq_len,1]
    d_mask_label [bs,seq_len,1]
    modified_expected_label: which kind of d_classify_output you want to produce higher rewards
    return:
    reward [batch_size,seq_len,1]
    (reward should be as large as possible for generators)
    """
    ew_nonlogit_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    # D explain reward
    # d_where_output more likely to be the reversed fake_mask_label
    # => higher reward
    r_batch_where=ew_nonlogit_cross_entropy(fake_mask_label,d_where_output)
    r_batch_where=tf.expand_dims(r_batch_where,2)

    # D classify reward
    
    if modified_expected_label == 1:
        # d_classify_output -> ones, then rewards higher
        unwanted_label = tf.zeros_like(d_classify_output)
    else:
        # d_classify_output -> zeros, then rewards higher
        unwanted_label = tf.ones_like(d_classify_output)

    r_batch_rf=ew_nonlogit_cross_entropy(unwanted_label,d_classify_output)
    r_batch_rf=tf.expand_dims(tf.expand_dims(r_batch_rf,1),2)
    r_batch_rf=tf.broadcast_to(r_batch_rf,[d_classify_output.shape[0],max_len,1])
    # objective funtion should be negative cross entropy
    rewards=tf.math.add(r_batch_where,r_batch_rf)

    # scale to [-1,1] in order to introduce positive and negative change to the loss gradient
#     rewards=my_normalization(rewards)
    return  rewards,r_batch_where,r_batch_rf# element-wise add
def G_mask_loss(rewards,g_mask_preds,fake_mask_label):
    """
    input rewards [bs,seq,1]
    g_mask_preds [bs,seq] 
    """
    # to prevent log(1)=0 
    g_mask_preds=tf.clip_by_value(g_mask_preds,1e-5,1.0-1e-5)
    reward_per_element=tf.multiply(rewards,tf.math.log(g_mask_preds))
    # non selected location set as zero
    fake_mask_label=tf.cast(fake_mask_label,tf.float32)
    fake_mask_label=tf.expand_dims(fake_mask_label,2)
    reward_per_element=tf.multiply(reward_per_element,fake_mask_label) # negative value
    reward_loss_per_line=- tf.math.reduce_sum(reward_per_element,1) 

    RL_loss=  tf.math.reduce_mean(reward_loss_per_line)
    
    return RL_loss

# from seqgan
def G_replace_loss(args,rewards,
                mask_indices,fake_mask_label,
                complete_ids,g_replace_output,
                inp_ids):
    """
    input:
    rewards [bs,seq,1]
    mask_indices [bs*ms,2]
    fake_mask_label [bs,seq_len] , 0/1 ,1 indicate masked.
    input_seq.shape=[bs,seq_len] batch of tokens
    input_emb [bs,seq,emb_dim]
    g_replace_output.shape=[bs,MASK_SIZE,vocab_size] batch of preds
    complete_out_emb batch of emb, [bs,MAX_SEQ_LEN,emb] maskword at padding emb is assigned zero
    use inner product to caculate the similarity of emb_dim
    """

    zero_preds=tf.zeros(shape=[fake_mask_label.shape[0],args.MAX_SEQUENCE_LENGTH,args.NB_WORDS],
                        dtype=g_replace_output.dtype)
    g_replace_output=tf.tensor_scatter_nd_update(tensor=zero_preds,indices=mask_indices,
                                    updates=tf.reshape(g_replace_output,[-1,args.NB_WORDS]))

    a= tf.one_hot(tf.cast(tf.reshape(complete_ids, [-1]),tf.int32),args.NB_WORDS, 1.0, 0.0)
    b= tf.cast(  
            tf.math.log(tf.clip_by_value(tf.reshape(g_replace_output, [-1,args.NB_WORDS]), 1e-5, 1.0-1e-5)),
            tf.float32)
    a=tf.reshape(a,[-1,args.NB_WORDS])
    b=tf.reshape(b,[-1,args.NB_WORDS])
    p_theta=tf.math.reduce_sum(tf.math.multiply(a,b),1)
    p_theta=tf.reshape(p_theta,[-1,args.MAX_SEQUENCE_LENGTH,1])
    
    RL_loss=-tf.math.multiply(rewards,p_theta)
    fake_mask_label=tf.expand_dims(tf.cast(fake_mask_label,tf.float32),2)
    RL_loss=tf.math.multiply(RL_loss,fake_mask_label)
    reward_loss_per_line=tf.math.reduce_sum(RL_loss,1)

    RL_loss=tf.math.reduce_mean(reward_loss_per_line)

    return RL_loss

def G_one_batch_train_step(args,LEX_GAN,inp): #, target
    '''
    inp = [rumor,non-rumor]
    '''
    if inp.shape[1]==2:
        m_inp,r_inp,n_inp=tf.concat(inp,axis=1),inp[:,0,:],inp[:,1,:]
        # inp = tf.reshape(inp,shape=[-1,inp.shape[-1]])
        # _, inp, _=tf.concat(inp,axis=1),inp[:,0,:],inp[:,1,:]
    else:
        print("Generator got wrong input")
        exit()

    def modified_inp_tokens(clean_inp,modified_size=args.MASK_SIZE,seq_max_len=args.MAX_SEQUENCE_LENGTH):
        # G MASK
        # [batch,seq_len]
        g_mask_preds=LEX_GAN.G_mask(clean_inp,training=True) 
        #  [bs,seq,1], [bs,seq,1], [bs*MASK_SIZE,2]
        fake_mask_label,_,mask_indices=Get_Mask(args,g_mask_preds,clean_inp,mask_size=modified_size)
        # [bs,seq,1]
        masked_inp=transform_input_with_is_missing_token(args,clean_inp,fake_mask_label)
        # G REPLACE
        # [bs,MASK_SIZE,vocab], [bs,MASK_SIZE] # dynamic size
        enc_hidden = LEX_GAN.G_replace.encoder.initialize_hidden_state()
        g_replace_preds, g_replace_preds_id=LEX_GAN.G_replace(args,masked_inp,enc_hidden,generate_size=args.MASK_SIZE,training=True)
        # complete generated sequence  #[bs,seq_len]
        complete_preds_id= replace_with_fake(args,clean_inp,g_replace_preds_id,mask_indices)
        complete_preds_id=tf.reshape(complete_preds_id,[clean_inp.shape[0],seq_max_len])

        return g_replace_preds, complete_preds_id, fake_mask_label, g_mask_preds, mask_indices

    def modify_selected_inp_and_update_Generator(args,selected_inp,modified_expected_label=1):
        with tf.GradientTape() as g_mask_tape,tf.GradientTape() as g_replace_tape:
            g_replace_preds, complete_preds_id, fake_mask_label, g_mask_preds, mask_indices = \
                modified_inp_tokens(selected_inp,modified_size=args.MASK_SIZE,seq_max_len=args.MAX_SEQUENCE_LENGTH)

            # update 2 Gs together
            g_real_fake_pred=LEX_GAN.D_classify(complete_preds_id,training=False)
            g_where_pred=LEX_GAN.D_where(complete_preds_id,training=False)
            # [bs,1]
            # d_classify_output,d_where_output,fake_mask_label
            g_rewards,_,_=calculate_rewards(g_real_fake_pred,g_where_pred,
                                    fake_mask_label=tf.expand_dims(fake_mask_label,2), 
                                    max_len=args.MAX_SEQUENCE_LENGTH,
                                    modified_expected_label=modified_expected_label)

            g_mask_loss=G_mask_loss(rewards=g_rewards,g_mask_preds=g_mask_preds
                                    ,fake_mask_label=fake_mask_label)
            g_replace_loss=G_replace_loss(args,rewards=g_rewards,
                                    mask_indices=mask_indices,fake_mask_label=fake_mask_label,
                                    complete_ids=complete_preds_id,g_replace_output=g_replace_preds,
                                    inp_ids=selected_inp)

        g_mask_gradients=g_mask_tape.gradient(g_mask_loss,LEX_GAN.G_mask.trainable_variables)
        LEX_GAN.g_mask_optimizer.apply_gradients(zip(g_mask_gradients,LEX_GAN.G_mask.trainable_variables))    

        g_replace_gradients = g_replace_tape.gradient(g_replace_loss, LEX_GAN.G_replace.trainable_variables)
        LEX_GAN.g_replace_optimizer.apply_gradients(zip(g_replace_gradients, LEX_GAN.G_replace.trainable_variables))

        return g_mask_loss,g_replace_loss

    # original version
    # g_mask_loss,g_replace_loss = modify_selected_inp_and_update_Generator(args,selected_inp=r_inp,modified_expected_label=1)

    mask_loss=[]
    replace_loss=[]

    gen_train_patterns = args.G_TRAIN_PATTERN.split('_') # default R-1
    for g_pattern in gen_train_patterns:
        inp_type, modification_direction = g_pattern.split('-')

        selected_inp = r_inp if inp_type == 'R' else n_inp
        modification_direction = int(modification_direction)
        g_mask_loss,g_replace_loss = modify_selected_inp_and_update_Generator( 
            args,selected_inp=selected_inp,
            modified_expected_label=modification_direction)

        mask_loss.append(g_mask_loss)
        replace_loss.append(g_replace_loss)

    return tf.math.reduce_mean(mask_loss), tf.math.reduce_mean(replace_loss)



# -------------------------Discriminator----------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
def reshape_and_cross_entropy(label,pred):
    logit_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # already sigmoid
    label=tf.reshape(label,[-1])
    pred=tf.reshape(pred,[-1])
    return logit_cross_entropy(label,pred)
def multi_class_cross_entropy(label,pred):
    # from softmax
    logit_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    label=tf.reshape(label,[-1,1])
    pred=tf.reshape(pred,[label.shape[0],-1]) # [batch, class_num] output_class_num should be 4
    # logging.error(f"sparsecat crosentropy\n label{label} pred {pred} ")
    return logit_cross_entropy(label,pred)



def D_where_loss(real_mask_pred, fake_mask_pred,fake_mask_label,inp,fake_inp):
    """
    [bs,seq,1]
    [bs,1]
    """
    if (inp==None) and (fake_inp==None):
        logging.error("Give at least one input sequence")
        exit()

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    real_where_loss=0
    fake_where_loss=0
    if inp is not None:
        real_sent_indices=tf.where(tf.cast(inp,tf.bool))# padding remove
        real_mask_pred=tf.gather_nd(real_mask_pred,real_sent_indices)
        real_mask_label=tf.zeros_like(real_mask_pred)
        real_where_loss = bce(real_mask_label,real_mask_pred)
    
    if fake_inp is not None:
        fake_sent_indices=tf.where(tf.cast(fake_inp,tf.bool))# padding remove
        fake_mask_pred=tf.gather_nd(fake_mask_pred,fake_sent_indices)
        fake_mask_label=tf.gather_nd(fake_mask_label,fake_sent_indices)
        fake_where_loss = bce(fake_mask_label, fake_mask_pred)

    where_loss= fake_where_loss + real_where_loss
    
    return where_loss

def D_one_batch_train_step(args, LEX_GAN, inp, d_classify_loss_function, D_classify_train=True,D_where_train=True): #, target
    """
    training phase: input [non-rumor,rumor]
    """
    if inp.shape[1]==2:
        m_inp,r_inp,n_inp= tf.reshape(inp,shape=[-1,inp.shape[-1]]),inp[:,0,:],inp[:,1,:]

    else:
        print("Discriminator got wrong input")
        exit()

    def modified_inp_tokens(clean_inp,modified_size=args.MASK_SIZE,seq_max_len=args.MAX_SEQUENCE_LENGTH):
        enc_hidden = LEX_GAN.G_replace.encoder.initialize_hidden_state()
        g_mask_preds=LEX_GAN.G_mask(clean_inp,training=False)

        fake_mask_label,_,mask_indices=Get_Mask(args,g_mask_preds,clean_inp,mask_size=modified_size)
        masked_inp=transform_input_with_is_missing_token(args,clean_inp,fake_mask_label)
        g_replace_preds, g_replace_preds_id=LEX_GAN.G_replace(args,masked_inp,enc_hidden,generate_size=modified_size,training=False)
        # complete generated sequence
        complete_preds_id= replace_with_fake(args,clean_inp,g_replace_preds_id,mask_indices)
        complete_preds_id=tf.reshape(complete_preds_id,[clean_inp.shape[0],seq_max_len])

        return complete_preds_id, fake_mask_label

    # for return 
    c_losses,w_losses,class_labels,class_preds,mask_labels,mask_preds = [None for i in range(6)]
    
    if D_classify_train:
        c_losses=[]
        class_preds=[]
        class_labels=[]
    if D_where_train:
        w_losses=[]
        mask_preds=[]
        mask_labels=[]

    def modify_selected_inp_and_update_Discriminator(args,selected_inp, 
            D_classify_train,D_where_train,
            modified=False,expected_label=1):

        d_classify_loss, classify_pred, classify_expect_label, d_where_loss, where_concat_preds, where_concat_labels = [None for i in range(6)]
        with tf.GradientTape() as tape, tf.GradientTape() as d_where_tape:
            if modified:
                complete_preds_id,fake_mask_label = modified_inp_tokens(
                    clean_inp=selected_inp,
                    modified_size=args.MASK_SIZE,
                    seq_max_len=args.MAX_SEQUENCE_LENGTH)
            # D_classify
            if D_classify_train:
                if modified:
                    classify_pred = LEX_GAN.D_classify(complete_preds_id, training=True)
                else:
                    classify_pred = LEX_GAN.D_classify(selected_inp, training=True)
                if str(args.D_CLASSIFY_OUTPUT).find('softmax') >= 0:
                    # 4 class label: [batch,1]; 4 class pred: [batch,4]
                    classify_expect_label = tf.ones(shape=[classify_pred.shape[0],1]) * expected_label
                else: 
                    # 2 class label: [batch,1]; 2 class preds: [batch,1]
                    classify_expect_label = tf.ones_like(classify_pred) * expected_label # expected_label could be 0,1,2,3
                # logging.error(f"D classifier batch output \n label:{classify_expect_label} pred{classify_pred}")
                # d_classify_loss= reshape_and_cross_entropy(classify_expect_label,classify_pred)   
                d_classify_loss = d_classify_loss_function(classify_expect_label,classify_pred)
                # cnn_l2_loss=tf.math.reduce_sum(LEX_GAN.D_classify.cnn.losses)
                # d_classify_loss+= cnn_l2_loss
            # D_where
            if D_where_train and modified:
                nr_mask_pred=LEX_GAN.D_where(selected_inp,training=True)
                fake_mask_pred=LEX_GAN.D_where(complete_preds_id,training=True)
                d_where_loss=D_where_loss(nr_mask_pred,fake_mask_pred,
                                    fake_mask_label,selected_inp,complete_preds_id)
        # update parameters
        if D_classify_train:
            d_classify_gradients=tape.gradient(d_classify_loss,LEX_GAN.D_classify.trainable_variables)
            LEX_GAN.d_classify_optimizer.apply_gradients(zip(d_classify_gradients,LEX_GAN.D_classify.trainable_variables))
        if D_where_train and modified:
            d_where_gradients=d_where_tape.gradient(d_where_loss,LEX_GAN.D_where.trainable_variables)
            LEX_GAN.d_where_optimizer.apply_gradients(zip(d_where_gradients,LEX_GAN.D_where.trainable_variables))

            # get shaped returns
            # compute the mask prediction from D_where
            nr_mask_pred,_,_=Get_Mask(args,nr_mask_pred,selected_inp,mask_size=args.MASK_SIZE)
            fake_mask_pred,_,_=Get_Mask(args,fake_mask_pred,complete_preds_id,mask_size=args.MASK_SIZE)
            
            # TODO: optimize operation here, maybe change to random length real sentences or abandon it.
            # delete predictions of padding locations for generated sentenece
            # make the real sentence the same short as generated sentence
            sent_indices=tf.where(tf.cast(selected_inp,tf.bool))
            nr_mask_pred=tf.gather_nd(nr_mask_pred,sent_indices)
            fake_mask_pred=tf.gather_nd(fake_mask_pred,sent_indices)   
            nr_mask_label=tf.zeros_like(nr_mask_pred)
            r_fake_mask_label=tf.gather_nd(fake_mask_label,sent_indices)     
            
            where_concat_preds = tf.concat([nr_mask_pred,fake_mask_pred],axis=0)
            where_concat_labels = tf.concat([nr_mask_label,r_fake_mask_label],axis=0)

        return d_classify_loss, classify_pred, classify_expect_label, d_where_loss, where_concat_preds, where_concat_labels

    for D_K_i in range(args.D_STEP_K):
        d_classify_train_patterns = args.D_CLASSIFY_TRAIN_PATTERN.split('_')
        for d_classify_pattern in d_classify_train_patterns:
            if len(d_classify_pattern.split('-')) == 2:
                is_modify = False
            elif len(d_classify_pattern.split('-')) == 3:
                is_modify = True
            else:
                logging.error("wrong pattern")
                exit()
            selected_inp = n_inp if d_classify_pattern.split('-')[0] == 'N' else r_inp
            expected_label = int(d_classify_pattern.split('-')[-1])
            
            d_classify_loss, classify_concat_preds, classify_concat_labels, \
                d_where_loss, where_concat_preds, where_concat_labels \
                    = modify_selected_inp_and_update_Discriminator(args,selected_inp, 
                        D_classify_train,D_where_train,
                        modified=is_modify,expected_label=expected_label)

            if D_classify_train:            
                c_losses.append(d_classify_loss)        
                class_preds.append(classify_concat_preds)
                class_labels.append(classify_concat_labels)
            if D_where_train and is_modify:
                w_losses.append(d_where_loss)
                mask_preds.append(where_concat_preds)
                mask_labels.append(where_concat_labels)
    
    # ----- end for ------

    if D_classify_train:
        c_losses=tf.math.reduce_mean(c_losses)
        class_labels=tf.reshape(tf.concat(class_labels,axis=0), [-1,])
        class_preds=tf.round(tf.reshape(tf.concat(class_preds,axis=0), [-1,]))
    if D_where_train:
        w_losses=tf.math.reduce_mean(w_losses)
        mask_preds=tf.reshape(tf.concat(mask_preds,axis=0), [-1,])
        mask_labels=tf.reshape(tf.concat(mask_labels,axis=0), [-1,])

    # batch loss and average accuratcy
    return c_losses,w_losses,class_labels,class_preds,mask_labels,mask_preds

# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

# TODO 增加 4 class label 的处理
# 1. 增加生成 Rm 和 Nm 并传入给 batch inference
# 2. 保存 生成的 text
# 3. 增加预先处理 Rm 和 Nm 的label

def discriminator_classifier_only_batch_inference(
    args,LEX_GAN,inp,inp_y,d_classify_loss_function,
    processor=None,case_study=False,logger=None):
    if inp.shape[1]==2:
        m_inp,r_inp,n_inp= tf.reshape(inp,shape=[-1,inp.shape[-1]]),inp[:,0,:],inp[:,1,:]
        m_inp_y,r_inp_y,n_inp_y= tf.reshape(inp_y,shape=[-1,1]),inp_y[:,0],inp_y[:,1]
    else:
        print("Discriminator got wrong input")
        exit()
    mix_class_pred =  LEX_GAN.D_classify(m_inp, training=False)
    d_classify_loss= d_classify_loss_function(m_inp_y,mix_class_pred )
    c_losses=d_classify_loss #tf.math.reduce_mean(c_losses)

    class_labels=tf.reshape(tf.concat([m_inp_y],axis=0), [-1,1])
    class_preds=tf.reshape(tf.concat([mix_class_pred],axis=0),[class_labels.shape[0],-1])

    return c_losses,class_labels,class_preds

def discriminator_where_only_batch_inference(
    args,LEX_GAN,inp,inp_y,
    processor=None,case_study=False,logger=None):
    if inp.shape[1]==2:
        m_inp,r_inp,n_inp= tf.reshape(inp,shape=[-1,inp.shape[-1]]),inp[:,0,:],inp[:,1,:]
        m_inp_y,r_inp_y,n_inp_y= tf.reshape(inp_y,shape=[-1,1]),inp_y[:,0],inp_y[:,1]
    else:
        print("Discriminator got wrong input")
        exit()

    mask_preds=[]
    mask_labels=[]

    enc_hidden = LEX_GAN.G_replace.encoder.initialize_hidden_state()
    g_mask_preds=LEX_GAN.G_mask(r_inp,training=False)
    fake_mask_label,_,mask_indices=Get_Mask(args,g_mask_preds,r_inp,mask_size=args.MASK_SIZE)
    masked_inp=transform_input_with_is_missing_token(args,r_inp,fake_mask_label)
    g_replace_preds, g_replace_preds_id=LEX_GAN.G_replace(args,masked_inp,enc_hidden,generate_size=args.MASK_SIZE,training=False)
    # complete generated sequence
    complete_preds_id= replace_with_fake(args,r_inp,g_replace_preds_id,mask_indices)
    complete_preds_id=tf.reshape(complete_preds_id,[r_inp.shape[0],args.MAX_SEQUENCE_LENGTH])
    sent_indices=tf.where(tf.cast(r_inp,tf.bool))



    # D_where
    real_mask_pred=LEX_GAN.D_where(r_inp,training=False)
    fake_mask_pred=LEX_GAN.D_where(complete_preds_id,training=False)

    d_where_loss=D_where_loss(
        real_mask_pred=real_mask_pred, 
        fake_mask_pred=fake_mask_pred,
        fake_mask_label=fake_mask_label,
        inp=r_inp,
        fake_inp=complete_preds_id
    )

    w_losses=d_where_loss #tf.math.reduce_mean(w_losses)

    nr_mask_pred,_,_=Get_Mask(args,real_mask_pred,r_inp,mask_size=args.MASK_SIZE)
    fake_mask_pred,_,_=Get_Mask(args,fake_mask_pred,complete_preds_id,mask_size=args.MASK_SIZE)
    
    nr_mask_pred=tf.gather_nd(nr_mask_pred,sent_indices)
    fake_mask_pred=tf.gather_nd(fake_mask_pred,sent_indices)         
    mask_preds.extend([nr_mask_pred,fake_mask_pred])

    nr_mask_label=tf.zeros_like(nr_mask_pred)
    mask_labels.extend([nr_mask_label,tf.gather_nd(fake_mask_label,sent_indices)])
   
    mask_preds=tf.reshape(tf.concat(mask_preds,axis=0), [-1,])
    mask_labels=tf.reshape(tf.concat(mask_labels,axis=0), [-1,])


    return w_losses,mask_labels,mask_preds

def discriminator_batch_inference(
    args,LEX_GAN,inp,inp_y,
    d_classify_loss_function,
    processor=None,case_study=False,logger=None,
    D_classify_val=True,D_where_val=True,
    out_f_modified_r=None,out_f_modified_n=None,
    balance_validation=True):
    
    def print_pair(prefix,tokens,labels):
        s_text = "{} input text:\n{}".format(prefix,processor._tokenizer.sequences_to_texts(tokens))
        s_label = "{} input label:\n{}".format(prefix,labels)
        return s_text + '\n' + s_label
    
    def print_case_study_format():
        
        # directly from dataset
        logger.info('original input:')
        logger.info(print_pair('non-rumor',n_inp.numpy(),n_inp_y.numpy()))
        logger.info(print_pair('rumor',r_inp.numpy(),r_inp_y.numpy()))

        # modified
        logger.info('G_mask selected locations:')
        logger.info(fake_mask_label.numpy())

        logger.info('G_replace output:')
        logger.info(print_pair('generated from non-rumor',complete_preds_id.numpy(),"all labeled as rumor"))
        
        logger.info("D_classifier predictions for non-modified text:")
        logger.info( mix_class_pred.numpy() )

        logger.info("D_explain predictions for non-modified text:")
        logger.info( real_mask_pred.numpy())

        logger.info("D_explain predictions for modified text:")
        logger.info( fake_mask_pred.numpy())

        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        return None

    def modified_inp_tokens(clean_inp,modified_size=args.MASK_SIZE,seq_max_len=args.MAX_SEQUENCE_LENGTH):
        # G MASK
        # [batch,seq_len]
        g_mask_preds=LEX_GAN.G_mask(clean_inp,training=False) 
        #  [bs,seq,1], [bs,seq,1], [bs*MASK_SIZE,2]
        fake_mask_label,_,mask_indices=Get_Mask(args,g_mask_preds,clean_inp,mask_size=modified_size)
        # [bs,seq,1]
        masked_inp=transform_input_with_is_missing_token(args,clean_inp,fake_mask_label)
        # G REPLACE
        # [bs,MASK_SIZE,vocab], [bs,MASK_SIZE] # dynamic size
        enc_hidden = LEX_GAN.G_replace.encoder.initialize_hidden_state()
        g_replace_preds, g_replace_preds_id=LEX_GAN.G_replace(args,masked_inp,enc_hidden,generate_size=args.MASK_SIZE,training=False)
        # complete generated sequence  #[bs,seq_len]
        complete_preds_id= replace_with_fake(args,clean_inp,g_replace_preds_id,mask_indices)
        complete_preds_id=tf.reshape(complete_preds_id,[clean_inp.shape[0],seq_max_len])

        return g_replace_preds, complete_preds_id, fake_mask_label, g_mask_preds, mask_indices

    
    c_losses,w_losses,class_labels,class_preds,mask_labels,mask_preds = [None for i in range(6)]
    
    
    m_inp=tf.reshape(inp,shape=[-1,inp.shape[-1]])
    m_inp_y=tf.reshape(inp_y,shape=[-1,1])
    num_rumor=tf.where(m_inp_y==0).shape[0]
    num_non_rumor=tf.where(m_inp_y==1).shape[0]    
    if inp.shape[1]==2 and balance_validation:
        r_inp,n_inp= inp[:,0,:],inp[:,1,:]
        r_inp_y,n_inp_y= inp_y[:,0],inp_y[:,1]
        # logger.debug(f"label debug {m_inp_y,n_inp_y,r_inp_y}")
    elif not balance_validation:
        if num_rumor > 0:
            r_inp=tf.gather(m_inp,tf.where(m_inp_y==0)[:,0])
            r_inp_y=tf.gather(m_inp_y,tf.where(m_inp_y==0)[:,0])
        if num_non_rumor > 0:
            n_inp=tf.gather(m_inp,tf.where(m_inp_y==1)[:,0])
            n_inp_y=tf.gather(m_inp_y,tf.where(m_inp_y==1)[:,0])
            
        if num_rumor>0 and num_non_rumor>0:
            m_inp = tf.concat([r_inp,n_inp],axis=0)
            m_inp_y = tf.concat([r_inp_y,n_inp_y],axis=0)
        else:
            m_inp=r_inp if num_rumor>0 else n_inp
            m_inp_y=r_inp_y if num_rumor>0 else n_inp_y
        
    else:
        print("Discriminator got wrong input shape, or balance problem")
        exit()

    
    
    # both need to modify r_inp & n_inp
    if D_where_val or str(args.D_CLASSIFY_OUTPUT).find('softmax') >= 0:
        if num_rumor>0:
            r_g_replace_preds, r_complete_preds_id, r_fake_mask_label, r_g_mask_preds, r_mask_indices = \
                modified_inp_tokens(r_inp,modified_size=args.MASK_SIZE,seq_max_len=args.MAX_SEQUENCE_LENGTH)
            if out_f_modified_r is not None:
                for modified_r in processor._tokenizer.sequences_to_texts(r_complete_preds_id.numpy()):
                    out_f_modified_r.write(modified_r.replace('<sos> ','').replace(' <eos>','')+'\n')
        if num_non_rumor>0:
            n_g_replace_preds, n_complete_preds_id, n_fake_mask_label, n_g_mask_preds, n_mask_indices = \
                modified_inp_tokens(n_inp,modified_size=args.MASK_SIZE,seq_max_len=args.MAX_SEQUENCE_LENGTH)
            if out_f_modified_n is not None:
                for modified_n in processor._tokenizer.sequences_to_texts(n_complete_preds_id.numpy()):
                    out_f_modified_n.write(modified_n.replace('<sos> ','').replace(' <eos>','')+'\n')
        if num_rumor>0 and num_non_rumor>0:
            modified_m_inp = tf.concat([r_complete_preds_id,n_complete_preds_id],axis=0)            
        else:
            modified_m_inp = r_complete_preds_id if num_rumor>0 else n_complete_preds_id

    if D_classify_val:
        class_preds=[]
        class_labels=[]

        # classify non-modified batch        
        if str(args.D_CLASSIFY_OUTPUT).find('softmax') >= 0:
            classify_inp = tf.concat([m_inp,modified_m_inp],axis=0)
            classify_inp_y = tf.concat([m_inp_y,m_inp_y+2],axis=0)
        else:
            classify_inp = m_inp
            classify_inp_y = m_inp_y
        
        classify_pred = LEX_GAN.D_classify(classify_inp, training=False)
        d_classify_loss = d_classify_loss_function(classify_inp_y, classify_pred)
        
        class_labels=tf.reshape(tf.concat(classify_inp_y,axis=0), [-1,1])
        class_preds=tf.reshape(tf.concat(classify_pred,axis=0),[class_labels.shape[0],-1])

        c_losses = d_classify_loss

    if D_where_val:
        mask_preds=[]
        mask_labels=[]
        
        real_mask_pred = LEX_GAN.D_where(m_inp,training=False)
        fake_mask_pred = LEX_GAN.D_where(modified_m_inp,training=False)
        if num_rumor>0 and num_non_rumor>0:
            fake_mask_label = tf.concat([r_fake_mask_label,n_fake_mask_label],axis=0)
        else:
            fake_mask_label = r_fake_mask_label if num_rumor>0 else n_fake_mask_label
        d_where_loss=D_where_loss(
            real_mask_pred=real_mask_pred, 
            fake_mask_pred=fake_mask_pred,
            fake_mask_label=fake_mask_label,
            inp=m_inp,
            fake_inp=modified_m_inp
        )
        w_losses=d_where_loss

        # construct formated d where label
        nr_mask_pred,_,_=Get_Mask(args,real_mask_pred,m_inp,mask_size=args.MASK_SIZE)
        fake_mask_pred,_,_=Get_Mask(args,fake_mask_pred,modified_m_inp,mask_size=args.MASK_SIZE)
        
        sent_indices=tf.where(tf.cast(m_inp,tf.bool))
        nr_mask_pred=tf.gather_nd(nr_mask_pred,sent_indices)
        fake_mask_pred=tf.gather_nd(fake_mask_pred,sent_indices)         
        mask_preds.extend([nr_mask_pred,fake_mask_pred])

        nr_mask_label=tf.zeros_like(nr_mask_pred)
        mask_labels.extend([nr_mask_label,tf.gather_nd(fake_mask_label,sent_indices)])

        mask_preds=tf.reshape(tf.concat(mask_preds,axis=0), [-1,])
        mask_labels=tf.reshape(tf.concat(mask_labels,axis=0), [-1,])

    if case_study:
        print_case_study_format()

    return c_losses,w_losses,class_labels,class_preds,mask_labels,mask_preds

#TODO 增加 4 class label 的处理 round or argmax
def discriminator_validation(
    args,LEX_GAN,valid_dataset, 
    processor,logger,output_modified_prefix=None,
    D_classify_val=True,D_where_val=True):

    if not(D_classify_val or D_where_val):
        logger.error("Need at least validate on one Discriminator")
        exit()

    if args.VAL_ONE_BATCH_CASE_STUDY:
        case_study_count=1
    # elif  args.ALL_BATCH_CASE_STUDY:
    #     TODO
    else:
        case_study_count=0

    if D_classify_val:
        d_classify_loss_v=[]
        d_classify_label_tuple=[[],[]] # only need from validation set: true, pred
    
    if D_where_val:
        d_where_loss_v=[]
        d_where_label_tuple=[[],[]] # only need from validation set: true, pred

    if str(args.D_CLASSIFY_OUTPUT).find('softmax') >= 0:
        d_classify_loss_function = multi_class_cross_entropy
    else:
        # binary
        d_classify_loss_function = reshape_and_cross_entropy
    
    if output_modified_prefix is not None:
        out_f_modified_r=open(os.path.join(args.OUTPUT_DIR,output_modified_prefix+'_modified_label_0_text.txt'),'w')
        out_f_modified_n=open(os.path.join(args.OUTPUT_DIR,output_modified_prefix+'_modified_label_1_text.txt'),'w') 
    else:
        out_f_modified_r=None
        out_f_modified_n=None

    for inp,inp_y in valid_dataset:

        if case_study_count > 0:
            case_study = True
            case_study_count = case_study_count - 1
        else:
            case_study = False

        # use condition to decide validation on which D
        d_classify_batch_loss,d_where_batch_loss,\
        class_labels,class_preds,mask_labels,mask_preds\
            =discriminator_batch_inference(
                args,LEX_GAN,inp,inp_y,
                d_classify_loss_function=d_classify_loss_function,
                processor=processor,case_study=case_study,logger=logger,
                D_classify_val=D_classify_val,D_where_val=D_where_val,
                out_f_modified_r=out_f_modified_r,out_f_modified_n=out_f_modified_n,
                balance_validation=args.IS_DEV_BALANCE,
            )

        if D_classify_val:
            d_classify_loss_v.append(d_classify_batch_loss)
            
            d_classify_label_tuple[0].append(class_labels)
            
            # label need to be round
            if str(args.D_CLASSIFY_OUTPUT).find('softmax') >= 0:
                class_preds = tf.math.argmax(class_preds,axis=1) # [batch,4] -> [batch,1]
            else:
                class_preds = tf.math.round(class_preds) # keep [batch,1], but from [probability] to [0 or 1]
            d_classify_label_tuple[1].append(class_preds) 

        if D_where_val:
            d_where_loss_v.append(d_where_batch_loss)
            d_where_label_tuple[0].append(mask_labels)
            d_where_label_tuple[1].append(mask_preds)

    if output_modified_prefix is not None:
        out_f_modified_r.close()
        out_f_modified_n.close()

    d_c_report,d_c_report_dict,d_w_report,d_w_report_dict = [None for i in range(4)]

    if D_classify_val:
        d_classify_loss_v = tf.math.reduce_mean(d_classify_loss_v).numpy()
        # convert to flat labels
        d_classify_label_tuple=[
            tf.concat(d_classify_label_tuple[0],axis=0).numpy(),
            tf.concat(d_classify_label_tuple[1],axis=0).numpy()
        ]
        # calculate metrics
        d_c_report,d_c_report_dict = get_D_classifier_report(d_classify_label_tuple)

    if D_where_val:
        d_where_loss_v = tf.math.reduce_mean(d_where_loss_v).numpy()
        d_where_label_tuple=[
            tf.concat(d_where_label_tuple[0],axis=0).numpy(),
            tf.concat(d_where_label_tuple[1],axis=0).numpy()
        ]
        # calculate metrics
        d_w_report,d_w_report_dict = get_Discriminator_where_report(d_where_label_tuple)
    
    return d_c_report,d_c_report_dict,d_w_report,d_w_report_dict

# first phase training

def train_first_phase(
    args,LEX_GAN,train_dataset,valid_dataset,
    max_global_epoch_step,max_val_macro_f1,
    processor,logger):
    
    # tensorboard initialization
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = args.OUTPUT_DIR + '/firtst_phase_logs/' + current_time + '/train'
    test_log_dir = args.OUTPUT_DIR + '/firtst_phase_logs/' + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    """
    the main purpose is to train the D_classify at this phase
    """
    early_stop_count=args.EARLY_STOP_EPOCH

    for epoch in range(max_global_epoch_step):
        best_perform_record=""
        logger.info("{} ".format(f"running epoch {epoch}"+'\n'))

        tf_logs={}

        g_mask_loss=[]
        g_replace_loss=[]

        for G_i in range(args.G_STEP):
            logger.info("{} ".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+"training generators at step {}..".format(G_i+1)+'\n')
            for inp,inp_y in train_dataset:
                # batch_count_idx=epoch*args.G_STEP*mix_train_size+G_i*args.G_STEP+batch_id
                g_mask_batch_loss,g_replace_batch_loss=G_one_batch_train_step(args,LEX_GAN,inp)
                g_mask_loss.append(g_mask_batch_loss)
                g_replace_loss.append(g_replace_batch_loss)

        tf_logs["g_mask_loss"] = tf.math.reduce_mean(g_mask_loss).numpy()
        tf_logs["g_replace_loss"] = tf.math.reduce_mean(g_replace_loss).numpy()

        d_where_loss=[]
        d_classify_loss=[]

        if str(args.D_CLASSIFY_OUTPUT).find('softmax') >= 0:
            d_classify_loss_function = multi_class_cross_entropy
        else:
            # binary
            d_classify_loss_function = reshape_and_cross_entropy

        for D_i in range(args.D_STEP):
            logger.info(f"training discriminators at step {D_i+1}..")
            for inp,inp_y in train_dataset:
                d_classify_batch_loss,d_where_batch_loss,\
                _,_,_,_=D_one_batch_train_step(
                    args,LEX_GAN, inp,
                    d_classify_loss_function=d_classify_loss_function,
                    D_classify_train=args.D_CLASSIFY_IS_TRAIN_FIRST_EPOCH,
                    D_where_train=args.D_WHERE_IS_TRAIN_FIRST_EPOCH)
                
                d_where_loss.append(d_where_batch_loss)
                d_classify_loss.append(d_classify_batch_loss)
            
            # if D_i < args.D_STEP:
            d_c_report,d_c_report_dict,_,_ = discriminator_validation(
                args,LEX_GAN,valid_dataset,processor,logger,output_modified_prefix=None,D_classify_val=True,D_where_val=False)
            logger.info(f"epoch {epoch} D_step {D_i+1} classifier report:\n{d_c_report}\n")
            d_classify_val_macro_f1 = d_c_report_dict['macro avg']['f1-score']
            if d_classify_val_macro_f1 > max_val_macro_f1:
                logger.info(f"epoch {epoch} D_step {D_i+1}: saving best model and report at D_classifier MAX VAL MACRO F1 {d_classify_val_macro_f1} ...\n")
                max_val_macro_f1=d_classify_val_macro_f1
                early_stop_count=args.EARLY_STOP_EPOCH
                LEX_GAN.ckpt_store(LEX_GAN.first_ckpt_prefix+"_best")
            else:
                early_stop_count-=1
                logger.info(f"val_macro_f1 not increasing for {args.EARLY_STOP_EPOCH - early_stop_count} epochs\n")
           

        
        tf_logs["d_classify_loss"] = tf.math.reduce_mean(d_classify_loss).numpy()

        # validation
        if args.D_WHERE_IS_TRAIN_FIRST_EPOCH:
            tf_logs["d_where_loss"] = tf.math.reduce_mean(d_where_loss).numpy()
            _,_,d_w_report,d_w_report_dict = discriminator_validation(
                args,LEX_GAN,valid_dataset,processor,logger,
                output_modified_prefix=None,D_classify_val=False,D_where_val=True)
            logger.info(f"locator report:\n{d_w_report}\n")
        
        # tensorboard log writting
        with train_summary_writer.as_default():
            for k,v in tf_logs.items():
                tf.summary.scalar(k, v, step=epoch)
        with test_summary_writer.as_default():
            tf.summary.scalar('d_classify_val_macro_f1', d_classify_val_macro_f1, step=epoch)
        
        
        
        
        s=f"finish trining epoch {epoch}\n"
        s= s + "training g_mask_loss: {}, g_replace_loss: {}\n".format(tf_logs["g_mask_loss"],tf_logs["g_replace_loss"])
        if args.D_WHERE_IS_TRAIN_FIRST_EPOCH:
            s+="d_where_loss: {}\n".format(tf_logs["d_where_loss"])
        if args.D_CLASSIFY_IS_TRAIN_FIRST_EPOCH:
            s+="d_classify_loss: {}\n".format(tf_logs["d_classify_loss"])
        logger.info(f"{s}\n")
      
        if early_stop_count <=0:
            break

    logging.info(f"restore best First Phase checkpoint tooutput modified train&dev text")
    LEX_GAN.ckpt_restore(LEX_GAN.first_ckpt_prefix+"_best")
    _,_,_,_ = discriminator_validation(
        args,LEX_GAN,train_dataset,processor,logger,
        output_modified_prefix=f"First_Phase_modified_mask_{args.MASK_SIZE}_train_dataset",D_classify_val=True,D_where_val=True)
    _,_,_,_ = discriminator_validation(
        args,LEX_GAN,valid_dataset,processor,logger,
        output_modified_prefix=f"First_Phase_modified_mask_{args.MASK_SIZE}_validation_dataset",D_classify_val=True,D_where_val=True)
    return max_val_macro_f1

def train_second_phase(args,LEX_GAN,epochs,EPOCHS, where_test_max_f1,training_mask_size):
    """
    the main purpose is to train the D_where at this phase
    """
    early_stop_count=args.EARLY_STOP_EPOCH
    for epoch in range(epochs,EPOCHS):
        print("running epoch ",epoch)
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        gmaskLoss=[]
        grpLoss=[]
        drfLoss=[]
        dwhLoss=[]
        for G_i in range(G_STEP):
            enum_train=enumerate(MIX_dataset.take(MIX_train_steps_per_epoch))
            print("training generators at step {}..".format(G_i+1))
            for (batch_id, (inp,targ)) in enum_train:
                maskBSloss,rpBSloss=G_one_batch_train_step(
                    inp,targ, enc_hidden,epoch*G_STEP*MIX_train_steps_per_epoch+G_i*G_STEP+batch_id) #tf.cast(inp,tf.int64), tf.cast(targ,tf.int64)
                gmaskLoss.append(maskBSloss)
                grpLoss.append(rpBSloss)
        gmaskLoss=np.mean(gmaskLoss)
        grpLoss=np.mean(grpLoss)

        for D_i in range(D_STEP):
            zip_train=zip(MIX_dataset.take(NR_train_steps_per_epoch),
                        N_dataset.take(NR_train_steps_per_epoch),
                        R_dataset.take(NR_train_steps_per_epoch))
            enum_train=enumerate(zip_train)#steps_per_epoch
            print("training discriminators at step {}..".format(D_i+1))
            for (batch_id, ((m_inp, m_targ),(n_inp,n_targ),(r_inp,r_targ))) in enum_train:
                drfBSloss,dwhBSloss,_,_,_,_= D_one_batch_second_phase(
                    [m_inp,n_inp,r_inp],[m_targ,n_targ,r_targ],
                    enc_hidden,epoch*D_STEP*NR_train_steps_per_epoch+D_i*D_STEP+batch_id,
                    True,training_mask_size=training_mask_size)
                drfLoss.append(drfBSloss)
                dwhLoss.append(dwhBSloss)
        drfLoss=np.mean(drfLoss)
        dwhLoss=np.mean(dwhLoss)
                    
        clear_output(wait=True) 

        b="network training performance at epoch{}:\n".format(epoch+1)
        b+="G-mask-loss".ljust(40)+"G-rplc-loss".ljust(40)+"\n"
        b+="{}".format(gmaskLoss).ljust(40)+"{}".format(grpLoss).ljust(40)+"\n"
        b+="D classify training loss:{}\n Dwhere training loss:{}\n".format(drfLoss,dwhLoss)
        # Test on the same image so that the progress of the model can be easily seen.
        a="Performance test for epoch: {}\n".format(epoch+1)
        drfLoss=[]
        dwhLoss=[]
        zip_test=zip(N_testset.take(NR_test_steps_per_epoch),R_testset.take(NR_test_steps_per_epoch))
        enum_test=enumerate(zip_test)
        dclassify_pred=[]
        dclassify_label=[]
        dwhere_pred=[]
        dwhere_label=[]
        #validation
        for (batch_id,( (n_inp,n_targ),(r_inp,r_targ))) in enum_test:
            drfBSloss,dwhBSloss,bs_c_pred,bs_c_label,bs_w_pred,bs_w_label= D_one_batch_second_phase(
                [n_inp,r_inp], [n_targ,r_targ], enc_hidden ,(epoch*NR_test_steps_per_epoch+batch_id),
                isTraining=False,training_mask_size=training_mask_size)
            drfLoss.append(drfBSloss)
            dwhLoss.append(dwhBSloss)
            dclassify_pred.append(bs_c_pred)
            dclassify_label.append(bs_c_label)
            dwhere_pred.append(bs_w_pred)
            dwhere_label.append(bs_w_label)
            # print(dclassify_pred)
            
        
        drfLoss=np.mean(drfLoss)
        dwhLoss=np.mean(dwhLoss)
        dclassify_pred=tf.concat(dclassify_pred,0).numpy().reshape((-1,1))
        dclassify_label=tf.concat(dclassify_label,0).numpy().reshape((-1,1))
        dwhere_pred=tf.concat(dwhere_pred,0).numpy().reshape((-1,1))
        dwhere_label=tf.concat(dwhere_label,0).numpy().reshape((-1,1))

        classify_val_f1=classification_report(dclassify_label,dclassify_pred,digits=6,output_dict=True)['macro avg']['f1-score']
        where_val_f1=classification_report(dwhere_label,dwhere_pred,digits=6,output_dict=True)['macro avg']['f1-score']

        b+="network VALIDATION performance at epoch{}:\n".format(epoch+1)
        b+="D classify validation loss:{}\n Dwhere validation loss:{}\n".format(drfLoss,dwhLoss)
        b+="D classify validation f1-score:{}\n Dwhere training mask size {} validation f1-score:{}\n".format(classify_val_f1,training_mask_size,where_val_f1)
        b+="D classify report\n"
        b+=classification_report(dclassify_label, dclassify_pred,digits=6)+"\n"
        b+="D where report on validation with training mask size{}\n".format(training_mask_size)
        b+=classification_report(dwhere_label, dwhere_pred,digits=6)+"\n"

        b+="D where report on normal mask size at test\n"
        rp,rp_dict=get_report_Dwhere(N_testset,MASK_SIZE)
        b+=rp
        where_test_f1=rp_dict['macro avg']['f1-score']
            
        if   where_test_f1>where_test_max_f1: # and classify_val_f1>classify_max_f1:
            b+="epoch {}: saving best model ..., max F1 {}".format(epoch+1,where_test_f1)+'\n'
            b+="D WHERE TEST MACRO F1:  {}\n".format(where_test_f1)
            b+="D classify validation f1-score:{}\n Dwhere validation f1-score:{}\n".format(classify_val_f1,where_val_f1)
            encoder.save_weights('./real_training_checkpoints/best_'+log_ver+'_second'+'/encoder')
            decoder.save_weights('./real_training_checkpoints/best_'+log_ver+'_second'+'/decoder')
            G_mask.save_weights('./real_training_checkpoints/best_'+log_ver+'_second'+'/gmask')
            D_han.save_weights('./real_training_checkpoints/best_'+log_ver+'_second'+'/drealfake')
            D_where.save_weights('./real_training_checkpoints/best_'+log_ver+'_second'+'/dwhere')
            saveOpt("best_"+log_ver)
            where_test_max_f1=where_test_f1
            mearly_stop_count=args.EARLY_STOP_EPOCH
        else:
            early_stop_count-=1

        print(b)
        print(a)

        with open(r"./test_performance_log_"+log_ver+".txt","a+") as f:
            f.write(b)
            f.write(a)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,time.time()-start))
        
        if early_stop_count <= 0:
            print("no increase for 3 epoch")
            break
        else:
            print("early stop count",early_stop_count)

    return where_test_max_f1