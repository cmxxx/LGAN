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
import argparse
import sys



import tensorflow as tf
print("GPU Available: ", tf.test.is_gpu_available())

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



# -------------- customized code -------------------
from data.text_proecssing import TextPreprocessor, process_text
# --------------__----------------------------------


# define the parameters
def get_arguments():


    parser = argparse.ArgumentParser()


    #"NN269_Donor",#"LIAR",#"PHEMEv5",'machine_fakenews'(fakenews_machine_data)
    parser.add_argument('--DATASET', type=str,default='machine_fakenews',
                           choices=['PHEMEv5', 'PHEMEv9','NN269_Accepter','NN269_Donor','machine_fakenews'],
                           help='name of the dataset')
    parser.add_argument('--WORKING_SPACE',type=str,default="./",
                          help='working path the main program locates')
    parser.add_argument('--VERSION', type=str,
                           help='name tag')
    parser.add_argument('--DATA_DIR', type=str,default="data",
                           help='path store the data')
    parser.add_argument('--OUTPUT_DIR', type=str,
                           help='path store the data')
    parser.add_argument('--GLOVE_DIM', type=int,default=50,
                           help='dim of pretrained embeddings')
    parser.add_argument('--PHEME_NUM_EVENT',type=int,
                help='set the event number when using pheme dataset')   
    
    # PHEMEv5 only:'charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting','sydneysiege'
    parser.add_argument('--DATA_EVENT_SPECIFY',type=str, 
                        choices=['NOT_HOLD-ONE',
                                 'charliehebdo', 'ebola-essien', 'ferguson',
                                 'germanwings-crash','gurlitt', 'ottawashooting',
                                 'prince-toronto', 'putinmissing','sydneysiege',
                                 'full_k40','mod_m_10'],
                        help='specify which dataset to use in the MIT dataset') 
    

    parser.add_argument('--IS_DELETE_RECORD',default=False, action="store_true",
                        help='delete used working space folder')
    parser.add_argument('--TRAIN_MODE',default=False, action="store_true",
                        help='suggests training or inference')
    parser.add_argument('--BATCH_SIZE',type=int,
                        help='...')
    parser.add_argument('--START_WORD',type=str,default="<sos>",
                        help='begining of each sequence')
    parser.add_argument('--END_WORD',type=str,default="<eos>",
                        help='begining of each sequence')
    parser.add_argument('--MASK_WORD',type=str,default="<replacemask>",
                        help='special word indicate mask position')
    parser.add_argument('--ENCODER_RNN_TYPE',type=str,default="GRU",
                        choices=["LSTM","GRU"],
                        help='specify the rnn-like layer in encoder')
    parser.add_argument('--MASK_SIZE',type=int,default=4,
                        help='number of blanks to fill')
    parser.add_argument('--MAX_SEQUENCE_LENGTH',type=int, default=50,
                        help='maximum length of each sentence')#17,#60, #50 for liar; 40 for pheme 
    parser.add_argument('--NUM_TEST',type=int,
                        help='specify test sample quantity when not hold-one-out training')
    parser.add_argument('--IS_OVER_SAMPLING',type=bool,
                        help='...')
    parser.add_argument('--IS_DEV_BALANCE',type=bool,default=True,
                    help='...')
    parser.add_argument('--EARLY_STOP_EPOCH',type=int,
                        help='stop training without increasing performance')
    parser.add_argument('--G_REPLACE_ENCODER_UNIT',type=list,
                        help='seq2seq-based G_replace model parameter')
    parser.add_argument('--G_REPLACE_DECODER_UNIT',type=list,
                        help='seq2seq-based G_replace model parameter')
    parser.add_argument('--G_WHERE_RNN_UNIT',type=int,
                        help='...')    
    parser.add_argument('--D_WHERE_RNN_UNIT',type=int,
                        help='...')        
    parser.add_argument('--D_CLASSIFY_CONV_UNIT', nargs='+', type=int,
                        help='seq2seq-based G_replace model parameter, list')
    parser.add_argument('--IS_LSTM_D_CLASSIFY',default=False, action="store_true",
                        help='')
    parser.add_argument('--D_CLASSIFY_OUTPUT',type=str, choices=['sigmoid','softmax_4', 'softmax_6','softmax_12'],
                        help='the output layer of d_classify')
    parser.add_argument('--G_R_LR',type=float,
                        help='g_repalce optimizer learning rate arguments') 
    parser.add_argument('--G_W_LR',type=float,
                        help='g_where optimizer learning rate arguments') 
    parser.add_argument('--D_C_LR',type=float,
                        help='d_classify optimizer learning rate arguments')
    parser.add_argument('--D_W_LR_1',type=float,
                        help='d_where optimizer learning rate at first phase')  
    parser.add_argument('--D_W_LR_2',type=float,
                        help='d_where optimizer learning rate at seconde phase')
    parser.add_argument('--D_C_L2_WEIGHT',type=float,
                        help='d_classify: the weight of l2 regularizer')
    parser.add_argument('--ADAM_BEAT_1',type=float,default=0.9,
                        help='optimizer arguments')
    parser.add_argument('--G_STEP',type=int,
                        help='step of generator training per epoch')
    parser.add_argument('--D_STEP',type=int,
                        help='step of discriminator training per epoch')
    parser.add_argument('--D_STEP_K',type=int,
                        help='step repeatition of discriminator training for each batch')  
    parser.add_argument('--FIRST_EPOCHS',type=int,
                        help='first phase training epoch')  
    parser.add_argument('--SECOND_EPOCHS',type=list,
                        help='second phase training epochs')
    parser.add_argument('--KEEP_LABEL',default=False, action="store_true",
                        help='only use GAN to do data augmentation, but not real/fake detector when multi-classification')
    parser.add_argument('--VAL_ONE_BATCH_CASE_STUDY',default=False, action="store_true",
                        help='print validation case')

    parser.add_argument('--D_CLASSIFY_IS_TRAIN_FIRST_EPOCH',default=False, action="store_true",
                        help='training switch')
    parser.add_argument('--D_WHERE_IS_TRAIN_FIRST_EPOCH',default=False, action="store_true",
                        help='training switch')
 
    parser.add_argument('--START_TOKEN',type=int,
                        help='NO DEAULT VALUE, dynamically defined token of start word')
    parser.add_argument('--END_TOKEN',type=int,
                        help='NO DEAULT VALUE, dynamically defined token of start word')
    parser.add_argument('--MASK_TOKEN',type=int,
                        help='NO DEAULT VALUE, dynamically defined token of mask word')  
    parser.add_argument('--NB_WORDS',type=int,
                        help='NO DEAULT VALUE, dynamically defined number of words')

    parser.add_argument('--D_CLASSIFY_PRETRARINED_PATH', 
                        type=str, 
                        help='previously trained model')
    

    parser.add_argument('--G_TRAIN_PATTERN', 
                        type=str,
                        default='R-1',
                        help='how to train generator, R-1 -> modify R to N, for multiple .. N-1_R-0')
    parser.add_argument('--D_CLASSIFY_TRAIN_PATTERN', 
                        type=str,
                        default='N-1_R-0_N-1_R-m-0',
                        help='how to train d classify, R-0 -> expect R as 0, for multiple .. N-1_R-0')

    parser.add_argument('--D_WHERE_TRAIN_PATTERN', 
                        type=str,
                        default='R-m-0',
                        help='how to train d where, only do with A-m-X ->  R-m-0 exists mask')

    parser.add_argument('--LOAD_MODIFIED_TRAIN_TEXT',type=str, default=None,
                        help='label as 2 3')
    parser.add_argument('--LOAD_MODIFIED_DEV_TEXT',type=str, default=None,
                        help='label as 2 3')

    parser.set_defaults(
        VERSION="v1",
        BATCH_SIZE=64,
        NUM_TEST=512, # num_test=512 if DATASET=="PHEMEv9" else 256, 512 for MFP
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
        G_STEP=1,
        D_STEP=3,
        D_STEP_K=3,
        FIRST_EPOCHS =500,
        SECOND_EPOCHS=[50,100,150],
        NB_WORDS=None,
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

 
    args.OUTPUT_DIR=os.path.join(args.WORKING_SPACE,"_".join([args.DATASET,args.ENCODER_RNN_TYPE,args.VERSION]))
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
        
        args.IS_DEV_BALANCE = (args.DATA_EVENT_SPECIFY == 'NOT_HOLD-ONE')
        logging.info(f" args.IS_DEV_BALANCE : {args.IS_DEV_BALANCE }")
        if os.path.isfile(f'{args.DATA_DIR}/{args.DATASET}_{args.MAX_SEQUENCE_LENGTH}_{args.DATA_EVENT_SPECIFY}_data_proccecced.pkl'):
            with open(f'{args.DATA_DIR}/{args.DATASET}_{args.MAX_SEQUENCE_LENGTH}_{args.DATA_EVENT_SPECIFY}_data_proccecced.pkl','rb') as f:
                train_token_set,dev_token_set,train_label,dev_label=pickle.load(f)
            with open(f'{args.DATA_DIR}/{args.DATASET}_{args.MAX_SEQUENCE_LENGTH}_{args.DATA_EVENT_SPECIFY}_proccesor.pkl',"rb") as f:
                processor=pickle.load(f)
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

    logging.info("{} ".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+'Size of Word Index:', len(processor._tokenizer.word_index)+1)
    if args.NB_WORDS is None:
        args.NB_WORDS=len(processor._tokenizer.word_index)+1 # len(processor._tokenizer.word_index)+1
    args.MASK_TOKEN=processor._tokenizer.word_index[args.MASK_WORD]
    args.START_TOKEN=processor._tokenizer.word_index[args.START_WORD]
    args.END_TOKEN=processor._tokenizer.word_index[args.END_WORD]
    logging.info("{} ".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+f"after processing: {args.NB_WORDS} {args.MASK_TOKEN} {args.START_TOKEN} {args.END_TOKEN}")
    logging.info("{} ".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+"Training/evaluation parameters %s", args)

    # create tensorflow dataset
    from data import text_to_tfdataset as ttt
    if args.DATASET=='PHEMEv5' or args.DATASET == 'PHEMEv9':
        train_dataset, dev_dataset = ttt.PHEME_tokens_to_dataset(args,train_token_set,dev_token_set,train_label,dev_label, event_name=args.DATA_EVENT_SPECIFY)
    elif args.DATASET=='machine_fakenews':
        train_dataset, dev_dataset = ttt.MFP_token_to_dataset(args,train_token_set,dev_token_set,train_label,dev_label,processor)
    else:
        logging.error(f"not support dataset {args.DATASET} yet")

    # -------------- trainer loading -------------------
    # first phase training for rumor/non-rumor classifier
    # second phase training for possible rumor-word locations
    if args.DATASET=='PHEMEv5' or args.DATASET == 'PHEMEv9':
        from training.MFP_trainer import train_first_phase,train_second_phase # unify API
    elif args.DATASET=='NN269_Accepter' or args.DATASET=='NN269_Donor':
        statement_set,label_set,processor=process_text(args)
        print("TODO")
    elif args.DATASET=='machine_fakenews':
        from training.MFP_trainer import train_first_phase,train_second_phase
    else:
        print("wrong dataset")
        exit()


    # load glove pretrained embeddings
    from utils.pretrained_emb_utils import load_golve_emb,load_saved_emb
    if os.path.isfile("{}/pretrained_emb_dim_{}_vocab_{}.pkl".format(args.OUTPUT_DIR,args.GLOVE_DIM,args.NB_WORDS)):
        pretrained_emb = load_saved_emb(args)
    else:
        pretrained_emb = load_golve_emb(args,processor)

    from models.models import LEX_GAN_model
    # create model
    LEX_GAN=LEX_GAN_model(args,pretrained_emb,event_name='mfp')

    # loading
    if args.D_CLASSIFY_PRETRARINED_PATH is not None:
        logger.info(f"loading D_classify from {args.D_CLASSIFY_PRETRARINED_PATH}")
        # need to call once before loading
        from training.MFP_trainer import discriminator_classifier_only_batch_inference,reshape_and_cross_entropy
        for inp,inp_y in train_dataset:
            _,_,_= discriminator_classifier_only_batch_inference(args,LEX_GAN,inp,inp_y,d_classify_loss_function=reshape_and_cross_entropy)
        LEX_GAN.D_classify.load_weights(args.D_CLASSIFY_PRETRARINED_PATH)
        # change the model structure if it need to be multiple classes
        if str(args.D_CLASSIFY_OUTPUT).find('softmax') >= 0:
            LEX_GAN.D_classify.ouput = LEX_GAN.D_classify.make_output_layer(args,force_make_2class=False)

    # training
    if args.TRAIN_MODE:
        # to train the Dclassify
        # 100 epoch should be enough
        logger.info(f"start first phase training")
        first_phase_maxf1 = train_first_phase(args,LEX_GAN,train_dataset,dev_dataset,
            max_global_epoch_step=args.FIRST_EPOCHS, 
            max_val_macro_f1=0.0, processor=processor,logger=logger)

        # to improve Dwhere with ascendent mask size
        best1=train_second_phase(args,LEX_GAN,epochs=0,EPOCHS=30,where_test_max_f1=0.0,training_mask_size=20)
        best2=train_second_phase(args,LEX_GAN,epochs=50,EPOCHS=100,where_test_max_f1=best1,training_mask_size=10)
        best3=train_second_phase(args,LEX_GAN,epochs=100,EPOCHS=150,where_test_max_f1=best2,training_mask_size=args.MASK_SIZE)
    else:
        from training.MFP_trainer import discriminator_validation
        logging.info(f"restore best First Phase checkpoint tooutput modified train&dev text")
        LEX_GAN.ckpt_restore(LEX_GAN.first_ckpt_prefix+"_best")
        _,_,_,_ = discriminator_validation(
            args,LEX_GAN,train_dataset,processor,logger,
            output_modified_prefix=f"First_Phase_modified_mask_{args.MASK_SIZE}_train_dataset",D_classify_val=True,D_where_val=True)
        _,_,_,_ = discriminator_validation(
            args,LEX_GAN,dev_dataset,processor,logger,
            output_modified_prefix=f"First_Phase_modified_mask_{args.MASK_SIZE}_validation_dataset",D_classify_val=True,D_where_val=True)


if __name__=="__main__":
    main()

