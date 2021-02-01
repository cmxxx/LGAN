import tensorflow as tf
import numpy as np 

def Get_Mask(args,preds,inp,mask_size,FIRST_PHASE=True):
    """
    intput:
    preds from g_mask [bs,seq_len,1]
    inp [bs,seq]

    2 returns:
    fake_mask_label [0 1 1 1] "1" indicate IS masked, shape [bs,seq,1]
    targ_represent [0 0 0 0] "0" indicate IS masked and it's for targets [bs,seq,1]
    """
    #idx from 0, size (bs,MASK_SIZE)
    if FIRST_PHASE:
        sampling_idx=tf.random.categorical(
            tf.reshape(preds,[-1,args.MAX_SEQUENCE_LENGTH]),mask_size)
    else:
        sampling_idx=tf.math.top_k(
            tf.reshape(preds,[-1,args.MAX_SEQUENCE_LENGTH] ), mask_size)[1]
    idx_one=[]
    bs=preds.shape[0]
    for j in range(bs):
        idx_one.extend([j for i in range(mask_size)])
    row_idx=tf.convert_to_tensor(np.array(idx_one)) # (bs*MASK_SIZE)
    num_idx=tf.cast(tf.reshape(sampling_idx,[-1]), tf.int64)#[ bs*ms]
    indices=tf.stack([row_idx,num_idx],axis=1)#(bs*ms, 2)
    
    labels=tf.zeros(shape=[bs,args.MAX_SEQUENCE_LENGTH,1])
    mask_value_broadcast=tf.ones([bs*mask_size,1],dtype=labels.dtype)#shape=(bs*MASK_SIZE, 1)

    labels_update=tf.scatter_nd(indices, mask_value_broadcast,labels.shape)

    update_mask = tf.scatter_nd(indices, tf.ones_like(num_idx, dtype=tf.bool), labels.shape[0:2])
    update_mask=tf.stack([update_mask],axis=2)

    fake_mask_label=tf.where(update_mask,labels_update,labels)

    # remove the mask label at padding
    # the mask indices is not handled, leave to the actual usage functions to deal with them
    mix_batch_size = args.BATCH_SIZE if inp.shape[0]==args.BATCH_SIZE else inp.shape[0]
    fake_mask_label=tf.reshape(fake_mask_label,[mix_batch_size,-1])
    not_padding_mask=tf.cast(inp,tf.bool)
    fake_mask_label=tf.where(not_padding_mask,fake_mask_label,tf.zeros_like(fake_mask_label)) # [bs,seq]
    
    # remove the mask label at <SOS> and <EOS>
    SOS_mask=tf.math.equal(inp, tf.constant(args.START_TOKEN))
    not_XOS_mask=tf.math.logical_not(SOS_mask)
    
    fake_mask_label=tf.where(not_XOS_mask,fake_mask_label,tf.zeros_like(fake_mask_label)).numpy()
    
    fake_mask_label[np.where(fake_mask_label>0)]=1.0
    fake_mask_label=tf.convert_to_tensor(fake_mask_label)
    targ_represent=fake_mask_label # useless

    return fake_mask_label,targ_represent,indices

def transform_input_with_is_missing_token(args,inp,fake_mask_label):
    replace_mask=tf.cast(fake_mask_label,tf.bool)
    mix_batch_size = args.BATCH_SIZE if inp.shape[0]==args.BATCH_SIZE else inp.shape[0]
    mask_tokens_matrix=tf.constant(
                                  args.MASK_TOKEN,
                                  dtype=tf.int32,
                                  shape=[mix_batch_size,args.MAX_SEQUENCE_LENGTH])
    return tf.where(replace_mask,mask_tokens_matrix,inp)

# change tokens to mask value
def replace_with_fake(args,inp,g_ids,mask_indices):
    """
    input:
    inp [batch,seq]
    g_ids [batch,MASK_SIZE]
    mask_indices[bs*ms,2]
    return:
    fake_seq [batch,seq]
    """

    g_ids=tf.reshape(tf.cast(g_ids,tf.int32),[-1])
    masked_inp= tf.tensor_scatter_nd_update(tensor=inp,indices=mask_indices,updates=g_ids)

    # the padding part should remain 0!
    not_padding_mask=tf.cast(inp,tf.bool)
    padding=tf.zeros(shape=masked_inp.shape,dtype=tf.int32)
    masked_inp=tf.where(not_padding_mask,masked_inp,padding)
    # XOS
    # remove the mask label at <SOS> and <EOS>
    SOS_mask=tf.math.equal(inp, tf.constant(args.START_TOKEN))
    not_XOS_mask=tf.math.logical_not(SOS_mask)
  
    masked_inp=tf.where(not_XOS_mask,masked_inp,inp)

    return masked_inp
    