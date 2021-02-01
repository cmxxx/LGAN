import numpy as np
import tensorflow as tf
import logging
from imblearn.over_sampling import SMOTE
# to get specific number of two class of data
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

def tokens_to_dataset(args,pdatatweets,plabel,event_name="NOT_HOLD-ONE"):
    
    if event_name =="NOT_HOLD-ONE":
        # original split: random
        pdata_train, pdata_test, plabel_train, plabel_test = my_split(pdatatweets,plabel,args.NUM_TEST,args.NUM_TEST)
    else:
        # split only one event
        p9_event = ['charliehebdo', 'ebola-essien', 'ferguson', 'germanwings-crash','gurlitt', 'ottawashooting', 'prince-toronto', 'putinmissing','sydneysiege']
        assert event_name in p9_event
        pdata_train, pdata_test, plabel_train, plabel_test=split_by_event(pdatatweets,plabel,pevent_label,event_name)

    print("total rumors in testset: ",sum(plabel_test==0))
    print("total non-rumors in testset: ",sum(plabel_test==1))
    
    # oversampling
    if args.IS_OVER_SAMPLING:
        sm = SMOTE(random_state=42)
        print("hold out event: ",event_name) 
        print("SMOTE ON TRAIN:")
        print("Before OverSampling, counts of label '1': {}".format(sum(plabel_train==1)))
        print("Before OverSampling, counts of label '0': {} \n".format(sum(plabel_train==0)))
        X_train_res, y_train_res = sm.fit_sample(pdata_train,plabel_train)
        print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
        print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
        print("SMOTE ON TEST:")
        print("Before OverSampling, counts of label '1': {}".format(sum(plabel_test==1)))
        print("Before OverSampling, counts of label '0': {} \n".format(sum(plabel_test==0)))
        if event_name =="NOT_HOLD-ONE":
            X_test_res, y_test_res = sm.fit_sample(pdata_test,plabel_test)
        else:
            X_test_res, y_test_res = pdata_test,plabel_test
        print("After OverSampling, counts of label '1': {}".format(sum(y_test_res==1)))
        print("After OverSampling, counts of label '0': {}".format(sum(y_test_res==0)))
    else:
        X_train_res, y_train_res, X_test_res, y_test_res = pdata_train, plabel_train, pdata_test,plabel_test
    
    with open('{}/{}_data_proccecced.pkl'.format(args.WORKING_SPACE,event_name),'wb') as f:
        pickle.dump([X_train_res, y_train_res,X_test_res, y_test_res],f)
    
    return X_train_res, y_train_res,X_test_res, y_test_res

def restore_proccesor_and_data(args,event_name):
    with open('{}/{}_data_proccecced.pkl'.format(args.WORKING_SPACE,event_name),'rb') as f:
        X_train_res, y_train_res,X_test_res, y_test_res=pickle.load(f)
    with open("{}/proccesor.pkl".format(args.WORKING_SPACE),"rb") as f:
        processor=pickle.load(f)
    return processor,X_train_res, y_train_res,X_test_res, y_test_res

# expand numpy array
def np_repeat_and_dump(npa,num):
    b=np.copy(npa)
    if npa.shape[0] <num:
        n_repeat=num//npa.shape[0] +1
        b=np.repeat(b,n_repeat,axis=0)
        idx = np.random.permutation(len(b))
        x = b[idx]
        return x
    else:
        return npa

def split_rumor_nonrumor(X_train_res, y_train_res,X_test_res, y_test_res):
    R_train=X_train_res[np.where(y_train_res==0)]
    N_train=X_train_res[np.where(y_train_res==1)]
    MIX_train=X_train_res
    N_test=X_test_res[np.where( y_test_res==1)]
    R_test=X_test_res[np.where( y_test_res==0)]
    return R_train,N_train,MIX_train,N_test,R_test
def get_tfdataset(args,input_train):
    dataset = tf.data.Dataset.from_tensor_slices(input_train).shuffle(input_train.shape[0])
    dataset = dataset.batch(args.BATCH_SIZE, drop_remainder=True)
    return dataset

def dumb_tfdataset_size(dataset):
    num_elements = 0
    for element in dataset:
        num_elements += 1
    return num_element



def convert_to_dataset(R_train,N_train,MIX_train,N_test,R_test,event_name):

    R_dataset=get_tfdataset(R_train.astype('int32'))
    N_dataset=get_tfdataset(N_train.astype('int32'))
    MIX_dataset=get_tfdataset(MIX_train.astype('int32'))
    R_testset=get_tfdataset(R_test.astype('int32'))
    # exclude 0 case bug
    if event_name!="ebola-essien": 
        N_testset=get_tfdataset(N_test.astype('int32'))
    else:
        N_testset=None
    return R_dataset,N_dataset,MIX_dataset,R_testset,N_testset


def PHEME_tokens_to_dataset(args,pdata_train, pdata_test, plabel_train, plabel_test, event_name="NOT_HOLD-ONE"):

    def convert(token_set,label):
        rumor_set=token_set[np.where(label==0)]
        nonrm_set=token_set[np.where(label==1)]
        np.random.shuffle(rumor_set)
        np.random.shuffle(nonrm_set)

        if rumor_set.shape[0] != nonrm_set.shape[0]:
            logging.error("rumor and non-rumor data not balanced!")
            exit()

        mix_set=[]
        lbl_pair_list=[]
        label_pair=np.array([0,1])
        for idx in range(rumor_set.shape[0]):
            mix_set.append([rumor_set[idx],nonrm_set[idx]])
            lbl_pair_list.append(label_pair)
        
        # TODO 添加  label 3 4 处理，因为VAE要用
        if np.any(np.isin(2,label)) or np.any(np.isin(3,label)):
            modified_rumor_set=token_set[np.where(label==2)]
            modified_nonrm_set=token_set[np.where(label==3)]
            np.random.shuffle(modified_rumor_set)
            np.random.shuffle(modified_nonrm_set)
            modified_label_pair=np.array([2,3])
            for idx in range(modified_rumor_set.shape[0]):
                mix_set.append([modified_rumor_set[idx],modified_nonrm_set[idx]])
                lbl_pair_list.append(modified_label_pair)            

        features_dataset = tf.data.Dataset.from_tensor_slices(mix_set)
        label_dataset = tf.data.Dataset.from_tensor_slices(lbl_pair_list)

        dataset = tf.data.Dataset.zip((features_dataset,label_dataset)).shuffle(len(mix_set))
        dataset = dataset.batch(args.BATCH_SIZE, drop_remainder=(args.LOAD_MODIFIED_DEV_TEXT is not None))# only VAE baseline need this

        return dataset
    
    logging.info(f"processing dataset of event {event_name}")
    train_dataset = convert(pdata_train, plabel_train)
    if args.IS_DEV_BALANCE:
        dev_dataset = convert(pdata_test, plabel_test)
    else:
        features_dataset=tf.data.Dataset.from_tensor_slices(pdata_test)
        label_dataset=tf.data.Dataset.from_tensor_slices(plabel_test)
        dev_dataset = tf.data.Dataset.zip((features_dataset,label_dataset)).shuffle(plabel_test.shape[0])
        dev_dataset = dev_dataset.batch(args.BATCH_SIZE, drop_remainder=(args.LOAD_MODIFIED_DEV_TEXT is not None))# only VAE baseline need this
        
    return train_dataset,dev_dataset


def MFP_token_to_dataset(args,train_token_set,dev_token_set,train_label,dev_label,processor):
    # each element in training set:
    # [rumor,non-rumor] + label [0, 1]

    # full_k40 already balanced
    def convert(token_set,label_list):

        token_set=np.asarray(token_set)
        label=np.asarray(label_list)
        #print(token_set[np.where(label==0)])
        rumor_set=token_set[np.where(label==0)]
        nonrm_set=token_set[np.where(label==1)]
        np.random.shuffle(rumor_set)
        np.random.shuffle(nonrm_set)

        mix_set=[]
        lbl_pair_list=[]
        label_pair=np.array([0,1])
        for idx in range(rumor_set.shape[0]):
            mix_set.append([rumor_set[idx],nonrm_set[idx]])
            lbl_pair_list.append(label_pair)

        if np.any(np.isin(2,label)) or np.any(np.isin(3,label)):
            modified_rumor_set=token_set[np.where(label==2)]
            modified_nonrm_set=token_set[np.where(label==3)]
            np.random.shuffle(modified_rumor_set)
            np.random.shuffle(modified_nonrm_set)
            modified_label_pair=np.array([2,3])
            for idx in range(modified_rumor_set.shape[0]):
                mix_set.append([modified_rumor_set[idx],modified_nonrm_set[idx]])
                lbl_pair_list.append(modified_label_pair)            

        features_dataset = tf.data.Dataset.from_tensor_slices(mix_set)
        label_dataset = tf.data.Dataset.from_tensor_slices(lbl_pair_list)

        dataset = tf.data.Dataset.zip((features_dataset,label_dataset)).shuffle(len(mix_set))
        dataset = dataset.batch(args.BATCH_SIZE, drop_remainder=(args.LOAD_MODIFIED_DEV_TEXT is not None))

        return dataset

    # print(type(train_token_set))
    # print(type(train_label))
    train_dataset = convert(train_token_set,train_label)
    dev_dataset = convert(dev_token_set,dev_label)

    return  train_dataset,dev_dataset

