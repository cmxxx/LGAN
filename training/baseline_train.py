import matplotlib
import datetime
import os
import shutil
import time
import numpy as np 
import pandas as pd
import pickle
import csv
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pickle
import argparse
import sys


import tensorflow as tf
print("GPU Available: ", tf.test.is_gpu_available())

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_arguments():


    parser = argparse.ArgumentParser()

    data_args = parser.add_argument_group('Data')
    data_args.add_argument('--DATASET', type=str,
                           choices=['PHEMEv5', 'PHEMEv9','LIAR','NN269_Accepter','NN269_Donor','machine_fakenews'],
                           help='name of the dataset')
    data_args.add_argument('--WORKING_SPACE',type=str,
                          help='working path the main program locates')
    data_args.add_argument('--VERSION', type=str,
                           help='name tag')
    data_args.add_argument('--DATA_DIR', type=str,
                           help='path store the data')
    data_args.add_argument('--OUTPUT_DIR', type=str,
                           help='path store the data')
    data_args.add_argument('--GLOVE_DIM', type=int,
                           help='dim of pretrained embeddings')
    data_args.add_argument('--PHEME_NUM_EVENT',type=int,
                help='set the event number when using pheme dataset')   
    data_args.add_argument('--DATA_EVENT_SPECIFY',type=str,
        help='specify which dataset to use in the MIT dataset') 
    train_args = parser.add_argument_group('Training')

    train_args.add_argument('--BASELINE_MODEL',type=str, choices=['CNN','LSTM', 'VAE-CNN','VAE-LSTM'],
                           help='specify model type')

    train_args.add_argument('--IS_DELETE_RECORD',type=bool,
                        help='delete used working space folder')
    train_args.add_argument('--TRAIN_MODE',type=bool,
                        help='suggests training or inference')
    train_args.add_argument('--BATCH_SIZE',type=int,
                        help='...')
    train_args.add_argument('--START_WORD',type=str,
                        help='begining of each sequence')
    train_args.add_argument('--END_WORD',type=str,
                        help='begining of each sequence')
    train_args.add_argument('--MASK_WORD',type=str,
                        help='special word indicate mask position')
    train_args.add_argument('--ENCODER_RNN_TYPE',type=str,
                    help='specify the rnn-like layer in encoder')
    train_args.add_argument('--MASK_SIZE',type=int,
                    help='number of blanks to fill')
    train_args.add_argument('--MAX_SEQUENCE_LENGTH',type=int,
                    help='maximum length of each sentence')

    train_args.add_argument('--NUM_TEST',type=int,
                    help='specify test sample quantity when not hold-one-out training')
    train_args.add_argument('--IS_OVER_SAMPLING',type=bool,
                    help='...')
    train_args.add_argument('--EARLY_STOP_EPOCH',type=int,
            help='stop training without increasing performance')

    train_args.add_argument('--G_REPLACE_ENCODER_UNIT',type=list,
            help='seq2seq-based G_replace model parameter')
    train_args.add_argument('--G_REPLACE_DECODER_UNIT',type=list,
            help='seq2seq-based G_replace model parameter')
    train_args.add_argument('--G_WHERE_RNN_UNIT',type=int,
            help='...')    
    train_args.add_argument('--D_WHERE_RNN_UNIT',type=int,
            help='...')        
    train_args.add_argument('--D_CLASSIFY_CONV_UNIT',type=list,
            help='seq2seq-based G_replace model parameter')
    train_args.add_argument('--D_CLASSIFY_OUTPUT',type=str, choices=['sigmoid','softmax_4', 'softmax_6','softmax_12'],
                           help='the output layer of d_classify')
    train_args.add_argument('--G_R_LR',type=float,
        help='g_repalce optimizer learning rate arguments') 
    train_args.add_argument('--G_W_LR',type=float,
        help='g_where optimizer learning rate arguments') 
    train_args.add_argument('--D_C_LR',type=float,
        help='d_classify optimizer learning rate arguments')
    train_args.add_argument('--D_W_LR_1',type=float,
        help='d_where optimizer learning rate at first phase')  
    train_args.add_argument('--D_W_LR_2',type=float,
        help='d_where optimizer learning rate at seconde phase')
    train_args.add_argument('--D_C_L2_WEIGHT',type=float,
        help='d_classify: the weight of l2 regularizer')
    train_args.add_argument('--ADAM_BEAT_1',type=float,
        help='optimizer arguments')
    train_args.add_argument('--G_STEP',type=int,
        help='step of generator training per epoch')
    train_args.add_argument('--D_STEP',type=int,
        help='step of discriminator training per epoch')
    train_args.add_argument('--D_STEP_K',type=int,
        help='step repeatition of discriminator training for each batch')  
    train_args.add_argument('--FRIST_EPOCHS',type=int,
        help='first phase training epoch')  
    train_args.add_argument('--SECOND_EPOCHS',type=list,
        help='second phase training epochs')
    train_args.add_argument('--KEEP_LABEL',type=bool,
        help='only use GAN to do data augmentation, but not real/fake detector when multi-classification')
    train_args.add_argument('--VAL_ONE_BATCH_CASE_STUDY',type=bool,
        help='print validation case')

 
    train_args.add_argument('--START_TOKEN',type=int,
                help='NO DEAULT VALUE, dynamically defined token of start word')
    train_args.add_argument('--END_TOKEN',type=int,
                help='NO DEAULT VALUE, dynamically defined token of start word')
    train_args.add_argument('--MASK_TOKEN',type=int,
                help='NO DEAULT VALUE, dynamically defined token of mask word')  
    train_args.add_argument('--NB_WORDS',type=int,
                help='NO DEAULT VALUE, dynamically defined number of words')  

    parser.add_argument('--LOAD_MODIFIED_TRAIN_TEXT',type=str, default=None,
        help='label as 2 3')
    parser.add_argument('--LOAD_MODIFIED_DEV_TEXT',type=str, default=None,
        help='label as 2 3')    

    parser.add_argument('--LOAD_DCLASSIFY_FROM_GAN',type=str, default=None,
        help='')    


    parser.set_defaults(
        DATASET='machine_fakenews',#"NN269_Donor",#"LIAR",#"PHEMEv5",'machine_fakenews'(fakenews_machine_data)
        WORKING_SPACE="./",
        VERSION="v1",
        DATA_DIR="./data",
        GLOVE_DIM=50,

        IS_DELETE_RECORD=False,
        TRAIN_MODE=True,
        BATCH_SIZE=64,
        START_WORD="<sos>",
        END_WORD="<eos>",
        MASK_WORD="<replacemask>",
        ENCODER_RNN_TYPE="GRU",
        MASK_SIZE=4,
        MAX_SEQUENCE_LENGTH = 50,#17,#60, #50 for liar; 40 for pheme 
        NUM_TEST=512, # num_test=512 if DATASET=="PHEMEv9" else 256, 512 for MFP
        IS_OVER_SAMPLING=True,
        EARLY_STOP_EPOCH=5,#5,
        G_REPLACE_ENCODER_UNIT=[32,32],
        G_REPLACE_DECODER_UNIT=[32,32],
        G_WHERE_RNN_UNIT=32,
        D_WHERE_RNN_UNIT=32,
        D_CLASSIFY_CONV_UNIT=[32,64],
        D_CLASSIFY_OUTPUT='softmax_4',
        G_R_LR=1e-4,
        G_W_LR=1e-4,
        D_C_LR=1e-4,
        D_W_LR_1=1e-4,
        D_W_LR_2=1e-3,
        D_C_L2_WEIGHT= 0.1, #0.1 #0.05 for hold-one-out
        ADAM_BEAT_1=0.9,
        G_STEP=1,
        D_STEP=3,
        D_STEP_K=3,
        FIRST_EPOCHS =500,
        SECOND_EPOCHS=[50,100,150],
        KEEP_LABEL=False,
        NB_WORDS=None,
        VAL_ONE_BATCH_CASE_STUDY=False,
        
    )
    args = parser.parse_args()

    return args

def set_args(args):
    if args.DATASET=='LIAR':
        args.GLOVE_DIM=200
        args.KEEP_LABEL=True
        args.MAX_SEQUENCE_LENGTH = 50
        args.VERSION="downgrad_smote"

    elif args.DATASET=='NN269_Accepter' or args.DATASET=='NN269_Donor':
        args.KEEP_LABEL=False
        args.MAX_SEQUENCE_LENGTH = 17
        args.VERSION="smote_four_class"

    args.OUTPUT_DIR=os.path.join(args.WORKING_SPACE,"_".join([args.DATASET,args.DATA_EVENT_SPECIFY,args.BASELINE_MODEL,args.VERSION]))
    if not os.path.exists(args.OUTPUT_DIR):
        os.makedirs(args.OUTPUT_DIR)

    if args.DATASET=="PHEMEv5":
        args.PHEME_NUM_EVENT=5
    elif args.DATASET=="PHEMEv9":
        args.PHEME_NUM_EVENT=9    
    else:
        # to suggest it's not
        args.PHEME_NUM_EVENT=-1
    return args



def main():
    args=get_arguments()
    args=set_args(args)

    logging.basicConfig(format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.DEBUG,datefmt='%Y-%m-%d%I:%M:%S %p')

    logger.info('Test logger')
    logging.info('logging info')
    logging.warning('logging warining')
    logging.error('logging error')

    # -------------- data loading -------------------
    from data.text_proecssing import process_text
    # process raw data to tokens
    if args.DATASET=='PHEMEv5' or args.DATASET == 'PHEMEv9':
        # pdatatweets,plabel,processor=process_text(args)
        if os.path.isfile(f'{args.DATA_DIR}/{args.DATASET}_{args.MAX_SEQUENCE_LENGTH}_{args.DATA_EVENT_SPECIFY}_data_proccecced.pkl'):
            with open(f'{args.DATA_DIR}/{args.DATASET}_{args.MAX_SEQUENCE_LENGTH}_{args.DATA_EVENT_SPECIFY}_data_proccecced.pkl','rb') as f:
                train_token_set,dev_token_set,train_label,dev_label=pickle.load(f)
            with open(f'{args.DATA_DIR}/{args.DATASET}_{args.MAX_SEQUENCE_LENGTH}_{args.DATA_EVENT_SPECIFY}_proccesor.pkl',"rb") as f:
                processor=pickle.load(f)

            modified_label_int = [2,3] if args.D_CLASSIFY_OUTPUT.find('softmax') >=0 else [0,1]
            if args.LOAD_MODIFIED_TRAIN_TEXT is not None:
                train_text_2=[]
                train_label_2=[]
                train_text_3=[]
                train_label_3=[]
                with open(args.LOAD_MODIFIED_TRAIN_TEXT.replace('index','0'),'r') as train_t:
                    for line in train_t:
                        train_text_2.append(line.strip())
                        train_label_2.append(modified_label_int[0])
                with open(args.LOAD_MODIFIED_TRAIN_TEXT.replace('index','1'),'r') as train_t:
                    for line in train_t:
                        train_text_3.append(line.strip())
                        train_label_3.append(modified_label_int[1])

                train_token_set = np.concatenate([train_token_set,processor.transform(train_text_2),processor.transform(train_text_3)],axis=0)
                train_label = np.concatenate([train_label,np.array(train_label_2),np.array(train_label_3)],axis=0)

            if args.LOAD_MODIFIED_DEV_TEXT is not None:
                dev_text_2=[]
                dev_label_2=[]
                dev_text_3=[]
                dev_label_3=[]
                with open(args.LOAD_MODIFIED_DEV_TEXT.replace('index','0'),'r') as dev_t:
                    for line in dev_t:
                        dev_text_2.append(line.strip())
                        dev_label_2.append(modified_label_int[0])
                with open(args.LOAD_MODIFIED_DEV_TEXT.replace('index','1'),'r') as dev_t:
                    for line in dev_t:
                        dev_text_3.append(line.strip())
                        dev_label_3.append(modified_label_int[1])

                dev_token_set = np.concatenate([dev_token_set,processor.transform(dev_text_2),processor.transform(dev_text_3)],axis=0)
                dev_label = np.concatenate([dev_label,np.array(dev_label_2),np.array(dev_label_3)],axis=0)
        else:
            train_token_set,dev_token_set,train_label,dev_label, processor = process_text(args)
    elif args.DATASET=='LIAR':
        statement_set,label_set,processor,le=process_text(args)
    elif args.DATASET=='NN269_Accepter' or args.DATASET=='NN269_Donor':
        statement_set,label_set,processor=process_text(args)
    elif args.DATASET=='machine_fakenews':
        train_token_set,dev_token_set,train_label,dev_label,processor = process_text(args) # [train_set,dev_set],[train_lable,dev_label]
    else:
        print("wrong dataset")
        exit()
        
    # load glove pretrained embeddings
    from utils.pretrained_emb_utils import load_golve_emb,load_saved_emb
    if os.path.isfile("{}/pretrained_emb_dim_{}_vocab_{}.pkl".format(args.OUTPUT_DIR,args.GLOVE_DIM,args.NB_WORDS)):
        pretrained_emb = load_saved_emb(args)
    else:
        pretrained_emb = load_golve_emb(args,processor)

    from training.baseline_trainer import trainer,VAE_discriminator_train,VAE_generator_pretrain

    # load model
    from models import baseline_models

    if args.BASELINE_MODEL.find('VAE') >= 0:
        from data import text_to_tfdataset as ttt
        if args.DATASET=='PHEMEv5' or args.DATASET == 'PHEMEv9':
            train_dataset, dev_dataset = ttt.PHEME_tokens_to_dataset(args,train_token_set,dev_token_set,train_label,dev_label, event_name="NOT_HOLD-ONE")
        elif args.DATASET=='machine_fakenews':
            train_dataset, dev_dataset = ttt.MFP_token_to_dataset(args,train_token_set,dev_token_set,train_label,dev_label,processor)
        else:
            logging.error(f"not support dataset {args.DATASET} yet")

        generator = baseline_models.CVAE(
            seq_length=args.MAX_SEQUENCE_LENGTH,
            latent_dim=16, vocab_size=args.NB_WORDS, 
            embedding_dim=args.GLOVE_DIM, 
            rnn_units=32, batch_size=args.BATCH_SIZE)

        if os.path.isfile(os.path.join(args.OUTPUT_DIR,'VAE_generator_weights_ckpt.index')):
            logging.info("restore generator weights")
            # activate model weights
            for inp,_ in train_dataset:
                m_inp,r_inp,n_inp= tf.reshape(inp,shape=[-1,inp.shape[-1]]),inp[:,0,:],inp[:,1,:]
                mean, logvar = generator.encode(m_inp)
                z = generator.reparameterize(mean,logvar)
                generated_images = generator.decode(z)
                break
            generator.load_weights(os.path.join(args.OUTPUT_DIR,'VAE_generator_weights_ckpt'))
        else:
            generator = VAE_generator_pretrain(args,generator=generator, dataset=train_dataset, epochs=1000)
            generator.save_weights(os.path.join(args.OUTPUT_DIR,'VAE_generator_weights_ckpt'))

        output_class_num = 4 if args.D_CLASSIFY_OUTPUT.find('softmax') >=0 else 1
        if args.BASELINE_MODEL.find('CNN') >= 0:
            model=baseline_models.VAE_make_cnn_discriminator_model(
                seq_length=args.MAX_SEQUENCE_LENGTH,
                latent_dim=16,vocab_size=args.NB_WORDS,
                embedding_dim=args.GLOVE_DIM,rnn_units=32,
                batch_size=args.BATCH_SIZE,
                output_class_num=output_class_num)
        elif args.BASELINE_MODEL.find('LSTM') >= 0:
            model=baseline_models.VAE_make_lstm_discriminator_model(
                seq_length=args.MAX_SEQUENCE_LENGTH,
                latent_dim=16,vocab_size=args.NB_WORDS,
                embedding_dim=args.GLOVE_DIM,rnn_units=32,
                batch_size=args.BATCH_SIZE,
                output_class_num=output_class_num)
        else:
            print("model TODO")
            exit()
        VAE_discriminator_train(args,discriminator=model,generator=generator, dataset=train_dataset, test=dev_dataset, epochs=100)
    else:
        if args.BASELINE_MODEL.find('CNN') >= 0:
            model = baseline_models.CNN_classify_model(args, 
                vocab_size=args.NB_WORDS,embedding_dim=args.GLOVE_DIM, 
                batch_sz=args.BATCH_SIZE,pre_embd=pretrained_emb, 
                conv_sizes=args.D_CLASSIFY_CONV_UNIT)
        elif args.BASELINE_MODEL.find('LSTM') >= 0:
            model = baseline_models.LSTM_classify_model(args, 
                vocab_size=args.NB_WORDS,embedding_dim=args.GLOVE_DIM, 
                batch_sz=args.BATCH_SIZE,pre_embd=pretrained_emb, 
                conv_sizes=[64,32])       
        else:
            print("model TODO")
            exit()
        trainer(args,model,train_token_set,dev_token_set,train_label,dev_label,processor,logger)

if __name__=="__main__":
    main()
