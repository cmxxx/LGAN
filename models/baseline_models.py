import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
import numpy as np
import os 
from tensorflow import keras

class CNN_classify_model(tf.keras.Model):
    def __init__(self, args, vocab_size, embedding_dim, batch_sz,pre_embd,conv_sizes=[64,128],force_make_2class=False):
        super(CNN_classify_model, self).__init__()
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

class LSTM_classify_model(tf.keras.Model):
    def __init__(self, args, vocab_size, embedding_dim, batch_sz,pre_embd,conv_sizes=[64,32],force_make_2class=False):
        super(LSTM_classify_model, self).__init__()
        self.batch_sz = batch_sz
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                  embeddings_initializer=tf.keras.initializers.Constant(pre_embd),trainable=True)
        self.lstm=self.make_discriminator_model(args,conv_sizes)
        self.ouput=self.make_output_layer(args,force_make_2class=force_make_2class)

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

class CVAE(tf.keras.Model):

    def __init__(self, seq_length, latent_dim, vocab_size, embedding_dim, rnn_units, batch_size):
        super(CVAE, self).__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.batch_size = batch_size * 2 # the input will be mix input of N & R, [batch*2,seq]
    #     self.weight_matrix = weight_matrix
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim,
                                      batch_input_shape=[self.batch_size, None],),
            tf.keras.layers.LSTM(self.rnn_units,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.LSTM(self.rnn_units,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(self.latent_dim + self.latent_dim,
                                  activation=tf.nn.relu)
        ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(
                    self.seq_length, self.latent_dim), batch_size=self.batch_size),
                #           tf.keras.layers.Dense(32,activation = tf.nn.tanh),
                #           tf.keras.layers.RepeatVector(vocab_size),
                tf.keras.layers.LSTM(self.rnn_units,
                                     return_sequences=True,
                                     stateful=True,
                                     recurrent_initializer='glorot_uniform'),
                tf.keras.layers.LSTM(self.rnn_units,
                                     return_sequences=True,
                                     stateful=True,
                                     recurrent_initializer='glorot_uniform'),
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(self.vocab_size, activation=tf.nn.relu)),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(BATCH_SIZE, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        #     print(x.shape)
        mean, logvar = tf.split(self.inference_net(
            x), num_or_size_splits=2, axis=2)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits


def VAE_make_lstm_discriminator_model(seq_length,latent_dim,vocab_size,embedding_dim,rnn_units, batch_size,output_class_num):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(seq_length,latent_dim),batch_size = batch_size*2),
        tf.keras.layers.LSTM(rnn_units,
                            return_sequences=True),
        tf.keras.layers.LSTM(16,
                            return_sequences=True),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8,activation=tf.nn.relu),
        tf.keras.layers.Dense(output_class_num)
      ]
    )
    return model

def VAE_make_cnn_discriminator_model(seq_length,latent_dim,vocab_size,embedding_dim,rnn_units, batch_size,output_class_num):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(seq_length,latent_dim),batch_size = batch_size*2))
    model.add(layers.Conv1D(32, ( 5),
                            strides=( 2), padding='same')) # kernel_regularizer=tf.keras.regularizers.l2(D_HAN_L2_WEIGHT), 
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv1D(64, ( 5), 
                             strides=( 2), padding='same'))# kernel_regularizer=tf.keras.regularizers.l2(D_HAN_L2_WEIGHT),
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(16,"relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(output_class_num))
    return model