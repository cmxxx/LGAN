import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow import keras
import numpy as np
import os 
import logging
# seq2seq
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, rnn_type="GRU"):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,trainable=True)
        self.rnn_type=rnn_type

        if self.rnn_type=="GRU":
            self.gru1=tf.keras.layers.GRU(self.enc_units[0],
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
            self.gru2=tf.keras.layers.GRU(self.enc_units[1],
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        elif self.rnn_type=="LSTM":
            self.lstm1=tf.keras.layers.LSTM(self.enc_units[0],
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
            self.lstm2=tf.keras.layers.LSTM(self.enc_units[1],
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        else:
            print("rnn type error")
            exit()
        
        self.initializer=tf.keras.initializers.Orthogonal()

    def call(self, x, hidden,training=True):
        # print(x.shape)
        # print(hidden.shape)
        x = self.embedding(x,training=training)
        if self.rnn_type=="GRU":
            output1, stateh1 = self.gru1(x,training=training) #, initial_state = hidden
            output2, stateh2 = self.gru2(output1,stateh1,training=training)   #, initial_state = stateh1
        elif self.rnn_type=="LSTM":
            output1, stateh1,statec1 = self.lstm1(x,training=training) #, initial_state = hidden
            output2, stateh2,statec2 = self.lstm2(output1,[stateh1,statec1],training=training)   #, initial_state = state1
        else:
            print("rnn type error")
            exit()

        return output2, stateh2

    def initialize_hidden_state(self):
        # return tf.zeros((self.batch_sz, self.enc_units))
        return self.initializer((self.batch_sz, self.enc_units[0]))

    def set_pretrain_embedding(self, vocab_size, embedding_dim, pre_embd):
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                 embeddings_initializer=tf.keras.initializers.Constant(pre_embd))
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units,activation='relu')
        self.W2 = tf.keras.layers.Dense(units,activation='relu')
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values,training=True):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values,training=training) + self.W2(hidden_with_time_axis,training=training)),training=training)

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz,rnn_type="GRU"):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,trainable=True)
        self.rnn_type=rnn_type

        if self.rnn_type=="GRU":
            self.gru1 = tf.keras.layers.GRU(self.dec_units[0],
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
            self.gru2 = tf.keras.layers.GRU(self.dec_units[1],
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer='glorot_uniform')
        elif self.rnn_type=="LSTM":
            self.lstm1=tf.keras.layers.LSTM(self.dec_units[0],
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
            self.lstm2=tf.keras.layers.LSTM(self.dec_units[1],
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        else:
            print("rnn type error")
            exit()        
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units[0])
        self.initializer=tf.keras.initializers.Orthogonal()
        
    def call(self, x, hidden, enc_output,training=True):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x,training=training)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        if self.rnn_type=="GRU":
            output1, stateh1 = self.gru1(x,training=training) #, initial_state = hidden
            output2, stateh2 = self.gru2(output1,stateh1,training=training)   #, initial_state = state1
        elif self.rnn_type=="LSTM":
            output1, stateh1,statec1 = self.lstm1(x,training=training) #, initial_state = hidden
            output2, stateh2,statec2 = self.lstm2(output1,[stateh1,statec1],training=training)   #, initial_state = state1
        else:
            print("rnn type error")
            exit()             

        output2 = tf.reshape(output2, (-1, output2.shape[2]))
        # output shape == (batch_size, vocab)
        output = self.fc(output2)

        return output, stateh2, attention_weights

    def set_pretrain_embedding(self, vocab_size, embedding_dim, pre_embd):
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                 embeddings_initializer=tf.keras.initializers.Constant(pre_embd))
        
class G_replace_model(tf.keras.Model):
    def __init__(self,args,pretrained_embeddings):
        super(G_replace_model,self).__init__()
        
        self.encoder = Encoder(args.NB_WORDS, args.GLOVE_DIM, args.G_REPLACE_ENCODER_UNIT, args.BATCH_SIZE)
        self.encoder.set_pretrain_embedding(args.NB_WORDS, args.GLOVE_DIM, pretrained_embeddings)
        self.decoder = Decoder(args.NB_WORDS, args.GLOVE_DIM, args.G_REPLACE_DECODER_UNIT, args.BATCH_SIZE)
        self.decoder.set_pretrain_embedding(args.NB_WORDS, args.GLOVE_DIM, pretrained_embeddings)
    
    def random_output(self,args,dec_id):
        a=list(range(args.NB_WORDS))
        a.remove(args.MASK_TOKEN)
        a.remove(args.START_TOKEN)
        a.remove(args.END_TOKEN)

        new_id=tf.constant(np.random.choice(a,dec_id.shape[0]))
        new_id=tf.cast(tf.reshape(new_id,dec_id.shape),dec_id.dtype)

        return new_id

    def replace_special_token(self,args,dec_predicted_id):
            # ------- code for skip special token --------
            # # if there is mask, run the loop again
            head_indices = tf.math.equal(dec_predicted_id,tf.constant(args.START_TOKEN,dtype=tf.int64))
            end_indices = tf.math.equal(dec_predicted_id,tf.constant(args.END_TOKEN,dtype=tf.int64))
            mask_indices = tf.math.equal(dec_predicted_id,tf.constant(args.MASK_TOKEN,dtype=tf.int64))

            special_token_indices= tf.math.logical_or(mask_indices, tf.math.logical_or(head_indices,end_indices)) 

            # there exists special token
            if tf.reduce_any(special_token_indices):
                
                random_output = self.random_output(args,dec_predicted_id)
                # if condition true, use x, else y
                dec_predicted_id = tf.where(condition=special_token_indices,x=random_output,y=dec_predicted_id)
                
            # ---------------------------------------------
            return dec_predicted_id

    def call(self,args,masked_inp,enc_hidden,generate_size,training=True):
        """
        Used for:
        2. While real training, only generate sequence of MASK_SIZE.
        return:
        preds [bs,generate_size,vocab_size]
        ids  [bs,generate_size]
        """


        enc_output, enc_hidden = self.encoder(masked_inp, enc_hidden,training=training)
        dec_input = tf.expand_dims([args.START_TOKEN ]* masked_inp.shape[0], 1)

        no_head=False
        no_mask=False
        # generate_size == mask/blank number
        # while not (no_mask and no_head):
        g_replace_preds=[]
        g_replace_preds_id=[]
        dec_hidden = enc_hidden
        # for t in range(0,generate_size):
        while generate_size>0:
            # passing enc_output to the decoder
            # dec output shape [bs,vocab_size]
            dec_predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output,training=training)

            # next input size: [bs,1,1]
            # if use argmax the generated output could be more stable
            # dec_predicted_id = tf.expand_dims(tf.argmax(dec_predictions,1),1) # easier for Discriminator, but worse generation quality # [batch,1]
            dec_predicted_id = tf.random.categorical(logits=dec_predictions,num_samples=1) # [batch,1]            
            dec_predicted_id = self.replace_special_token(args,dec_predicted_id)

            dec_input = dec_predicted_id
            generate_size-=1     
            g_replace_preds.append(tf.expand_dims(dec_predictions,1))
            g_replace_preds_id.append(tf.expand_dims(dec_predicted_id,1))       

        
        g_replace_preds=tf.concat(g_replace_preds,1) # [bs,seq_len,vocabsize]
        g_replace_preds_id=tf.concat(g_replace_preds_id,1) # [bs,seq_len]
        
        return  g_replace_preds, g_replace_preds_id


class D_classify_model(tf.keras.Model):
    def __init__(self, args, vocab_size, embedding_dim, batch_sz,pre_embd,conv_sizes=[64,128],force_make_2class=False):
        super(D_classify_model, self).__init__()
        self.batch_sz = batch_sz
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                  embeddings_initializer=tf.keras.initializers.Constant(pre_embd),trainable=True)
        self.cnn=self.make_discriminator_model(args,conv_sizes)
        self.ouput=self.make_output_layer(args,force_make_2class=force_make_2class)
    def call(self, x,training=True):
        x=self.embedding(x,training=training)
        x=self.cnn(x,training=training)
        x=self.ouput(x,training=training)
        return x
    def make_discriminator_model(self,args,conv_sizes):
        model = tf.keras.Sequential()
        # TODO: adjust kernel size maybe?
        for i in range(len(conv_sizes)):
            model.add(layers.Conv1D(conv_sizes[i], (5),
                                    kernel_regularizer=tf.keras.regularizers.l2(args.D_C_L2_WEIGHT), strides=( 2), padding='same'))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.2))

        # model.add(layers.Conv1D(conv_sizes[1], (5), 
        #                         kernel_regularizer=tf.keras.regularizers.l2(args.D_C_L2_WEIGHT), strides=( 2), padding='same'))
        # model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(0.5))
        return model
    def make_output_layer(self,args, force_make_2class=False):
        model = tf.keras.Sequential()
        if args.D_CLASSIFY_OUTPUT=='sigmoid' or force_make_2class:
            # if need to load 2-class pretrained classifier, need to be the same structure,
            # so force to construct the same output layer
            model.add(layers.Flatten())
            model.add(layers.Dense(1,'sigmoid'))

        elif str(args.D_CLASSIFY_OUTPUT).find('softmax') >= 0:

            activation, output_class_num = str(args.D_CLASSIFY_OUTPUT).split('_')
            output_class_num = int(output_class_num)

            model.add(layers.Dense(32,'relu'))
            model.add(layers.Flatten())
            model.add(layers.Dense(output_class_num,'softmax'))
        else:
            print("wrong d classify output layer")
            exit()

        return model

 
class D_classify_LSTM_model(tf.keras.Model):
    def __init__(self, args, vocab_size, embedding_dim, batch_sz,pre_embd,conv_sizes=[64,32],force_make_2class=False):
        super(D_classify_LSTM_model, self).__init__()
        self.batch_sz = batch_sz
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                  embeddings_initializer=tf.keras.initializers.Constant(pre_embd),trainable=True)
        self.lstm=self.make_discriminator_model(args,conv_sizes)
        self.ouput=self.make_output_layer(args,force_make_2class=force_make_2class) # cant' use name 'output'

    def call(self, x,training=True):
        x=self.embedding(x,training=training)
        x=self.lstm(x,training=training)
        x=self.ouput(x,training=training)
        return x
    def make_discriminator_model(self,args,conv_sizes):
        model = tf.keras.Sequential()
        # TODO: adjust kernel size maybe?
        for i in range(len(conv_sizes)):
            model.add(layers.LSTM(conv_sizes[i],return_sequences= i+1 <len(conv_sizes)))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.2))
        
        model.add(keras.layers.Dense(8, activation=tf.nn.relu))
       
        return model

    def make_output_layer(self,args, force_make_2class=False):
        model = tf.keras.Sequential()

        if args.D_CLASSIFY_OUTPUT=='sigmoid' or force_make_2class:
            model.add(keras.layers.Dense(1, 'sigmoid'))
        elif str(args.D_CLASSIFY_OUTPUT).find('softmax') >= 0:
            activation, output_class_num = str(args.D_CLASSIFY_OUTPUT).split('_')
            output_class_num = int(output_class_num)
            model.add(layers.Dense(32,'relu'))
            model.add(layers.Dense(output_class_num,'softmax'))
        else:
            print("wrong d classify output layer")
            exit()

        return model    

# for G where and D where
def make_rnn_mask_model(vocab_size, embedding_dim, pre_embd,rnn_units, batch_size,last_act="sigmoid"):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                # batch_input_shape=[batch_size*2, None],
                                weights=[pre_embd],),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_units,
                                                        return_sequences=True,
                                                        # stateful=True,
                                                        recurrent_initializer='glorot_uniform')),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_units,
                                                    return_sequences=True,
                                                    # stateful=True,
                                                    recurrent_initializer='glorot_uniform')),
        tf.keras.layers.Dense(16,activation='relu'),
        tf.keras.layers.Dense(units=1,activation=last_act)
    ])
    return model

class LEX_GAN_model(tf.keras.Model):
    '''
    model
    optimizer
    checkpoint
    '''
    def __init__(self,args,pretrained_embeddings,event_name):
        super(LEX_GAN_model, self).__init__()
    
        # G replace
        self.G_replace=G_replace_model(args,pretrained_embeddings)
        # G where
        self.G_mask=make_rnn_mask_model(args.NB_WORDS, args.GLOVE_DIM,pretrained_embeddings,args.G_WHERE_RNN_UNIT, args.BATCH_SIZE)
        # D classify
        force_make_2class = args.D_CLASSIFY_PRETRARINED_PATH is not None
        
        
        if args.IS_LSTM_D_CLASSIFY:
            self.D_classify=D_classify_LSTM_model(args,args.NB_WORDS, args.GLOVE_DIM, 
                args.BATCH_SIZE,pretrained_embeddings,
                conv_sizes=args.D_CLASSIFY_CONV_UNIT, # [64,32]
                force_make_2class=force_make_2class)
        else:
            self.D_classify=D_classify_model(args,args.NB_WORDS, args.GLOVE_DIM, 
                args.BATCH_SIZE,pretrained_embeddings,
                conv_sizes=args.D_CLASSIFY_CONV_UNIT,
                force_make_2class=force_make_2class)
            
        #D where
        self.D_where=make_rnn_mask_model(args.NB_WORDS, args.GLOVE_DIM, pretrained_embeddings,args.D_WHERE_RNN_UNIT, args.BATCH_SIZE)

        self.g_mask_optimizer = tf.keras.optimizers.Adam(learning_rate=args.G_W_LR ,beta_1=args.ADAM_BEAT_1)
        self.g_replace_optimizer = tf.keras.optimizers.Adam(learning_rate=args.G_R_LR ,beta_1=args.ADAM_BEAT_1)
        self.d_classify_optimizer = tf.keras.optimizers.Adam(learning_rate=args.D_C_LR ,beta_1=args.ADAM_BEAT_1)
        # need to reset learning rate for the second phase training
        self.d_where_optimizer = tf.keras.optimizers.Adam(learning_rate=args.D_W_LR_1,beta_1=args.ADAM_BEAT_1)


        #checkpoints
        self.first_ckpt_dir = "{}/{}_first_phase_checkpoints".format(args.OUTPUT_DIR,event_name)
        self.first_ckpt_prefix = os.path.join(self.first_ckpt_dir, "ckpt")

        self.second_ckpt_dir="{}/{}_second_phase_checkpoints".format(args.OUTPUT_DIR,event_name)
        self.second_ckpt_prefix = os.path.join(self.second_ckpt_dir,"ckpt")


    def ckpt_restore(self,ckpt_prefix):
        self.D_classify.load_weights(ckpt_prefix+"_D_classify")
        self.D_where.load_weights(ckpt_prefix+"_D_where")
        self.G_replace.load_weights(ckpt_prefix+"_G_replace")
        self.G_mask.load_weights(ckpt_prefix+"_G_mask")
    
    def ckpt_store(self,ckpt_prefix):
        self.D_classify.save_weights(ckpt_prefix+"_D_classify")
        self.D_where.save_weights(ckpt_prefix+"_D_where")
        self.G_replace.save_weights(ckpt_prefix+"_G_replace")
        self.G_mask.save_weights(ckpt_prefix+"_G_mask")

