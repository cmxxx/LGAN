import numpy as np
import re
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.sequence import pad_sequences
from os import path
from imblearn.over_sampling import SMOTE
import logging
import pickle

# convert text to tokens
class TextPreprocessor(object):
    
    def __init__(self, args):
        print('init processor')
        self._vocab_size = args.NB_WORDS
        self._max_sequence_length = args.MAX_SEQUENCE_LENGTH

        self._tokenizer = text.Tokenizer(num_words= args.NB_WORDS ,
                                        filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', 
                                        lower=True, char_level=False) # save all vocabularies

        self._pad_sequences = pad_sequences
        self._re_sub = re.sub

        self._head_word=args.START_WORD
        self._end_word=args.END_WORD

    def _clean_line(self, text):
        text = self._re_sub(r"http\S+", "<url>", text)
        text = self._re_sub(r"@[A-Za-z0-9]+", "<user>", text)
        text = self._re_sub(r"#[A-Za-z0-9]+", "", text)
        text = text.lower()
        text = text.strip()
        return text

    def fit(self,args, text_list):        

        text_list = [self._clean_line(txt) for txt in text_list]
        self._tokenizer.fit_on_texts(text_list)
        special_token = args.START_WORD + ' ' + args.END_WORD + ' '+ args.MASK_WORD
        self._tokenizer.fit_on_texts([special_token for i in range(len(text_list)*50)])
        self._head_token,self._end_token,self._mask_token = self._tokenizer.texts_to_sequences([args.START_WORD,args.END_WORD,args.MASK_WORD])
        self._head_token = self._head_token[0]
        self._end_token = self._end_token[0]
        self._mask_token = self._mask_token[0]
        print(self._tokenizer.texts_to_sequences(['the','it','to']))
        print(self._head_token,self._end_token,self._mask_token)


    def _token_sequences_add_head(self,token_list):    
        for idx in range(len(token_list)):
            token_list[idx] = [self._head_token] + token_list[idx]
        return token_list
    def _token_matrix_add_end(self,token_matrix):
        #print(token_matrix.shape)
        end_token = np.ones([token_matrix.shape[0],1],dtype='int32')
        #print(end_token.shape)
        end_token = end_token * self._end_token
        #print(end_token.shape)
        token_matrix=np.concatenate((token_matrix,end_token),axis=1)
        #print(token_matrix.shape)
        return token_matrix

    def transform(self, text_list):

        # Transform text to sequence of integers

        text_list = [self._clean_line(txt) for txt in text_list]
        seq = self._token_sequences_add_head(self._tokenizer.texts_to_sequences(text_list))

        padded_text_sequence = self._pad_sequences(
            sequences=seq,
            maxlen=self._max_sequence_length - 1, #  leave a place of <EOS>
            dtype='int32',
            padding='pre',
            truncating='post', # keep the start word
        )

        padded_text_sequence = self._token_matrix_add_end(padded_text_sequence)
        print(self._tokenizer.sequences_to_texts([padded_text_sequence[0]]))
        return padded_text_sequence#,target_tokens

def my_split(data,label,zero_num,one_num):
    idx = np.random.permutation(len(data))
    x,y = data[idx], label[idx]
    zero_idx=np.where(y==0)[0] # extract np data from tuple
    one_idx=np.where(y==1)[0]

    zero_num=zero_idx.shape[0]-zero_num
    one_num=one_idx.shape[0]-one_num
    x_train=np.concatenate((x[zero_idx[:zero_num]],x[one_idx[:one_num]]),axis=0)
    y_train=np.concatenate((y[zero_idx[:zero_num]],y[one_idx[:one_num]]),axis=0)
    x_test=np.concatenate((x[zero_idx[zero_num:]],x[one_idx[one_num:]]),axis=0)
    y_test=np.concatenate((y[zero_idx[zero_num:]],y[one_idx[one_num:]]),axis=0)

    return x_train,x_test,y_train,y_test

def split_by_event(data,label,event_label,select_event):
    # select_event as test set
    train_idx=np.where(event_label!=select_event)
    test_idx=np.where(event_label==select_event)

    x_train=data[train_idx]
    x_test=data[test_idx]
    y_train=label[train_idx]
    y_test=label[test_idx]
    
    return x_train,x_test,y_train,y_test

def process_text(args):
    print("loading and processing dataset {}".format(args.DATASET))
    processor = TextPreprocessor(args)

    if args.DATASET=='PHEMEv5' or  args.DATASET == 'PHEMEv9':
        prefix = "pheme_9" if args.DATASET=="PHEMEv9" else "pheme_5"
        
        pdatatweets=np.load(path.join(args.DATA_DIR,prefix+"_data.npy"),allow_pickle=True)
        plabel=np.load(path.join(args.DATA_DIR,prefix+"_label.npy"),allow_pickle=True)
        
        train_idx=np.copy(plabel)
        # reverse the labels
        # 1 as non-rumor 0 as rumor
        plabel[np.where(train_idx==1)]=0
        plabel[np.where(train_idx==0)]=1
        processor.fit(args,pdatatweets)

        pdatatweets=processor.transform(pdatatweets)
        # pdata_train, pdata_test, plabel_train, plabel_test = my_split(pdatatweets,plabel,args.NUM_TEST,args.NUM_TEST)
        if args.DATA_EVENT_SPECIFY =="NOT_HOLD-ONE":
            train_token_set,dev_token_set,train_label,dev_label = my_split(pdatatweets,plabel,args.NUM_TEST,args.NUM_TEST)
        else:
            pevent_label=np.load(path.join(args.DATA_DIR,prefix+"_event_label.npy"),allow_pickle=True)
            
            train_token_set,dev_token_set,train_label,dev_label = split_by_event(
                                                                        data=pdatatweets,
                                                                        label=plabel,
                                                                        event_label=pevent_label,
                                                                        select_event=args.DATA_EVENT_SPECIFY
                                                                    )
        logging.info(f"total rumors:non-rumors in trainingset is {sum(train_label==0)}:{sum(train_label==1)} ")            
        logging.info(f"total rumors:non-rumors in testset is {sum(dev_label==0)}:{sum(dev_label==1)} ")
    
        # oversampling
        if args.IS_OVER_SAMPLING:
            logging.info("hold out event:",args.DATA_EVENT_SPECIFY) 
            logging.info(f"SMOTE ON TRAIN:\nBefore OverSampling, counts of label '1': {sum(train_label==1)}\nBefore OverSampling, counts of label '0': {sum(train_label==0)} ")
            logging.info(f"SMOTE ON TEST:\nBefore OverSampling, counts of label '1': {sum(dev_label==1)}\nBefore OverSampling, counts of label '0': {sum(dev_label==0)} ")
            
            sm = SMOTE(random_state=42)

            train_token_set,train_label = sm.fit_sample(train_token_set,train_label)
            if args.DATA_EVENT_SPECIFY =="NOT_HOLD-ONE":
                dev_token_set,dev_label = sm.fit_sample(dev_token_set,dev_label)
            else:
                dev_token_set,dev_label = dev_token_set,dev_label

            logging.info(f"After OverSampling, counts of label '1': {sum(train_label==1)}\nAfter OverSampling, counts of label '0': {sum(train_label==0)}")
            logging.info(f"After OverSampling, counts of label '1': {sum(dev_label==1)}\nAfter OverSampling, counts of label '0': {sum(dev_label==0)}")
        
        else:
            logging.error("need to do oversampling or downsampling")
        
        with open(f'{args.DATA_DIR}/{args.DATASET}_{args.MAX_SEQUENCE_LENGTH}_{args.DATA_EVENT_SPECIFY}_data_proccecced.pkl','wb') as f:
            pickle.dump([train_token_set, dev_token_set, train_label, dev_label],f)
        with open(f'{args.DATA_DIR}/{args.DATASET}_{args.MAX_SEQUENCE_LENGTH}_{args.DATA_EVENT_SPECIFY}_proccesor.pkl','wb') as f:
            pickle.dump(processor,f)       

        # return pdata_train, pdata_test, plabel_train, plabel_test, processor
        return train_token_set,dev_token_set,train_label,dev_label, processor

    elif args.DATASET=='LIAR':
        filepaths=['{}/LIAR_train.tsv'.format(args.DATA_DIR),'{}/LIAR_valid.tsv'.format(args.DATA_DIR),'{}/LIAR_test.tsv'.format(args.DATA_DIR)]
        # two list of list [train,valid,test]
        statement_set=[]
        label_set=[]
        sm = SMOTE(random_state=42)
        le = preprocessing.LabelEncoder()
        if args.KEEP_LABEL:
            le.fit(['barely-true','false','half-true','mostly-true','pants-fire','true'])
        else:
            le.fit(['barely-true','false','half-true','mostly-true','pants-fire','true',
                    'z-barely-true','z-false','z-half-true','z-mostly-true','z-pants-fire','z-true'])
        for filepath in filepaths:
            statement=[]
            label=[]
            testfile = open(filepath, newline='')
            testreader = csv.reader(testfile, delimiter='\t', quotechar='|')
            for row in testreader:
                label.append(row[1])
                statement.append(row[2])
            testfile.close()
            processor.fit(statement)
            # str -> int
            statement=processor.transform(statement)
            label=le.transform(label)
            if filepath=='{}/LIAR_train.tsv'.format(args.DATA_DIR):
                statement,label = sm.fit_sample(statement,label)
                
            statement_set.append(statement) 
            label_set.append(label)

        return statement_set,label_set,processor,le
    elif args.DATASET=='NN269_Accepter' or args.DATASET=='NN269_Donor':
        prefix=args.DATASET
        train_data,train_label,test_data,test_label=\
            np.load(path.join(args.DATA_DIR,prefix+"_train_test_data_label.npy"),allow_pickle=True)
        train_data=[" ".join(x) for x in train_data]
        test_data=[" ".join(x) for x in test_data]
        
        processor.fit(train_data)

        train_data=processor.transform(train_data)
        test_data=processor.transform(test_data)

        sm = SMOTE(random_state=42)
        train_data, train_label= sm.fit_sample(train_data, train_label)

        statement_set=[train_data,test_data]
        label_set=[train_label,test_label]
        
        return statement_set,label_set,processor

    elif args.DATASET=='machine_fakenews':
        # currently use full_k40
        prefix=path.join(args.DATA_DIR,args.DATA_EVENT_SPECIFY)

        train_text_out_path=prefix+"_train_text.tsv"
        train_label_out_path=prefix+"_train_label.tsv"
        dev_text_out_path=prefix+"_dev_text.tsv"
        dev_label_out_path=prefix+"_dev_label.tsv"

        train_text=[]
        train_label=[]
        dev_text=[]
        dev_label=[]
        with open(train_text_out_path,'r') as train_t, open(train_label_out_path,'r') as train_l, \
            open(dev_text_out_path,'r') as dev_t, open(dev_label_out_path,'r') as dev_l :
            for line in train_t:
                train_text.append(line.strip())
            for line in train_l:
                train_label.append(int(line.strip()))
            for line in dev_t:
                dev_text.append(line.strip())
            for line in dev_l:
                dev_label.append(int(line.strip()))            
        
        processor.fit(args, [args.START_WORD,args.END_WORD,args.MASK_WORD]+train_text+dev_text)
#        processor.fit(dev_text)
#        print(dev_text[0])
        train_token_set = processor.transform(train_text)
        dev_token_set = processor.transform(dev_text)

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
            train_label = train_label + train_label_2 + train_label_3

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
            dev_label = dev_label + dev_label_2 + dev_label_3           


        return train_token_set,dev_token_set,train_label,dev_label,processor

    else:
        print("wrong dataset type")
        exit()

