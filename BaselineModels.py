import numpy as np
import pandas as pd

from keras.layers import merge, TimeDistributed
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional,GRU, GlobalMaxPool1D, Dropout, Conv1D, MaxPooling1D, Flatten,Convolution1D, Reshape
from keras.regularizers import L1L2
from keras.initializers import RandomNormal
from keras.layers.merge import Concatenate
from keras.models import model_from_json
import data_helpers as dh
from data_helpers import alphabet


vocabsize=len(alphabet)+2
###################
'''
this script contains several baselines for text classification
'''
class General():
    def __init__(self,):
       self.model=None
       # Training parameters
       self.batch_size = None
       self.num_epochs = None
       # Prepossessing parameters
       self.sequence_length = None
       self.vocab_size = None  ## changed to fit data size
       self.LoadedModel=None
       self.Model=None
       self.ExternalEmbeddingModel = None
       self.EmbeddingType=None

    def set_etxrernal_embedding(self,ModelFile,ModelType):
        self.ExternalEmbeddingModel=ModelFile
        self.EmbeddingType=ModelType

    def set_training_paramters(self,batch_size,num_epochs):
        self.batch_size=batch_size
        self.num_epochs=num_epochs

    def set_processing_parameters(self,sequence_length,vocab_size):
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size

    def train_model(self,Model,X_train,Y_train,X_valid,Y_valid):
        Model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=self.num_epochs, batch_size=self.batch_size)

    def Evaluate_model(self,Model,X_test,Y_test):
        score=Model.evaluate(X_test,Y_test,verbose=0)
        return score

    def save_model(self,ModelFileName,Model):
        print("Saving model in directory:")
        JsonModel = Model.to_json()
        with open('models/' + ModelFileName + ".json", "w") as json_file:
            json_file.write(JsonModel)
        Model.save_weights('models/' + ModelFileName + ".h")
        print('model saved in directory')

    def Load_model(self,ModelFileName):
        print("Loading Model from directory!")

        JsonFile = open(ModelFileName+".json",'r')

        # Load Json file
        LoadedModel = model_from_json(JsonFile)

        # Load weights
        LoadedModel.load_weights(ModelFileName+".h5")
        LoadedModel.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return LoadedModel

    #def Return_preds(self,Model,X_test):


class cnn_kim(General): ##inherits General
    '''
    CNNfor text classification based on kim 2014
    works for both static and non-static
    different is that network is initialized with RandomNormal distribution
    of small standard deviation
    '''
    def __init__(self,cnn_rand=True,STATIC=False,ExternalEmbeddingModel=None,EmbeddingType=None,n_symbols=None,wordmap=None):
        # Model hyperparameters
        self.embedding_dim=300##
        self.filter_sizes = (3, 8)
        self.num_filters = 10
        self.hidden_dims=100
        self.dropout_prob=(0.5,0.8)
        self.loss='categorical_crossentropy'
        self.optimizer= 'rmsprop'
        self.l1_reg=0
        self.l2_reg=3 ##according to kim14
        self.std=0.05 ## standard deviation
        # Training Parameters
        self.set_training_paramters(batch_size=64,num_epochs=10)
        self.set_processing_parameters(sequence_length=30,vocab_size=vocabsize) ## changed to fit short text

        # Defining Model Layers
        if cnn_rand:
            ##Embedding Layer Randomly initialized
            embedding_layer=Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size)
            Classes = dh.read_labels()
            n_classes = len(Classes)

        else:
            ## Use pretrained model
            #n_symbols, wordmap = dh.get_word_map_num_symbols()
            self.set_etxrernal_embedding(ExternalEmbeddingModel,ModelType=EmbeddingType)
            if self.EmbeddingType == "skipgram" or self.EmbeddingType == "CBOW":
               vecDic = dh.GetVecDicFromGensim(self.ExternalEmbeddingModel)
            elif self.EmbeddingType == "fastText":
               vecDic = dh.load_fasttext(self.ExternalEmbeddingModel)
            Classes = dh.read_labels()
            n_classes = len(Classes)
            ## Define Embedding Layer
            embedding_weights = dh.GetEmbeddingWeights(embedding_dim=300, n_symbols=n_symbols, wordmap=wordmap,
                                                       vecDic=vecDic)
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=n_symbols, trainable=STATIC)
            embedding_layer.build((None,))  # if you don't do this, the next step won't work
            embedding_layer.set_weights([embedding_weights])

        Sequence_in = Input(shape=(self.sequence_length,), dtype='int32')
        embedding_seq = embedding_layer(Sequence_in)
        x = Dropout(self.dropout_prob[0])(embedding_seq)
        ## define Core Convultional Layers
        conv_blocks = []
        for sz in self.filter_sizes:
            conv = Convolution1D(filters=self.num_filters,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(x)
            conv =  MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        x = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        x = Dropout(self.dropout_prob[1])(x)
        x = Dense(self.hidden_dims, activation="relu",kernel_initializer=RandomNormal(stddev=self.std),
                         kernel_regularizer=L1L2(l1=self.l1_reg,l2=self.l2_reg))(x)
        preds = Dense(n_classes, activation='softmax')(x)
        ## return graph model
        model = Model(Sequence_in, preds)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.model=model


class CrepeCNN(General): ## Todo



    def __init__(self,crepe_rand=True,STATIC=False,ExternalEmbeddingModel=None,EmbeddingType=None,n_symbols=None,wordmap=None,vocabsize=None,maxseq=None,embedding_dim=None):
     '''
     Deep CNN for text classification based on Lecun15
     '''

     self.embedding_dim = embedding_dim
     self.filter_kernels = [7, 7, 3, 3, 3, 3]
     self.nb_filters = 256
     self.batch_size = 64
     self.nb_epochs = 10
     self.std = 0.05
     self.dropout_prob = (0.5, 0.8)
     self.hidden_dim = 300
     self.loss = 'categorical_crossentropy'
     self.optimizer = 'rmsprop'
     ''' Set Training Parameters'''

     self.set_training_paramters(batch_size=self.batch_size, num_epochs=self.nb_epochs)
     self.set_processing_parameters(sequence_length=maxseq, vocab_size=vocabsize)

     Classes = dh.read_labels()
     n_classes = len(Classes)


     if crepe_rand:
         ##Embedding Layer Randomly initialized
         embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size)

     else:
         ## Use pretrained model
         # n_symbols, wordmap = dh.get_word_map_num_symbols()
         self.set_etxrernal_embedding(ExternalEmbeddingModel, ModelType=EmbeddingType)
         if self.EmbeddingType == "skipgram" or self.EmbeddingType == "CBOW":
             vecDic = dh.GetVecDicFromGensim(self.ExternalEmbeddingModel)
         elif self.EmbeddingType == "fastText":
             vecDic = dh.load_fasttext(self.ExternalEmbeddingModel)
         Classes = dh.read_labels()
         n_classes = len(Classes)
         ## Define Embedding Layer
         embedding_weights = dh.GetEmbeddingWeights(embedding_dim=300, n_symbols=n_symbols, wordmap=wordmap,
                                                    vecDic=vecDic)
         embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=n_symbols, trainable=STATIC)
         embedding_layer.build((None,))  # if you don't do this, the next step won't work
         embedding_layer.set_weights([embedding_weights])


     SequenceIn=Input(shape=(self.sequence_length,), dtype='int32')
     embedding_layer=embedding_layer(SequenceIn)
     x = Dropout(self.dropout_prob[0])(embedding_layer)
     x = Convolution1D(filters=self.nb_filters,kernel_size=self.filter_kernels[0],padding='valid',activation='relu')(x)
     #x = MaxPooling1D(pool_size=3)(x)
     x = Convolution1D(filters=self.nb_filters, kernel_size=self.filter_kernels[1], padding='valid', activation='relu')(x)
     #x = MaxPooling1D(pool_size=4)(x)
     x = Convolution1D(filters=self.nb_filters, kernel_size=self.filter_kernels[2], padding='valid', activation='relu')(x)
     x = Convolution1D(filters=self.nb_filters, kernel_size=self.filter_kernels[3], padding='valid', activation='relu')(x)
     x = Convolution1D(filters=self.nb_filters, kernel_size=self.filter_kernels[4], padding='valid', activation='relu')(x)
     x = Convolution1D(filters=self.nb_filters, kernel_size=self.filter_kernels[5], padding='valid', activation='relu')(x)
     x = MaxPooling1D(pool_size=3)(x)
     x = Flatten() (x)
     x = Dense(self.hidden_dim,activation='relu')(x)
     x = Dropout(self.dropout_prob[1])(x)
     x = Dense(self.hidden_dim, activation='relu')(x)
     x = Dropout(self.dropout_prob[1])(x)
     preds = Dense(n_classes, activation='softmax')(x)
     ## return graph model
     model = Model(SequenceIn, preds)
     model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
     self.model = model



class cnn_char(General): ##inherits General
    ## Todo
    '''
    CNNfor text classification based on kim 2014
    works for both static and non-static
    different is that network is initialized with RandomNormal distribution
    of small standard deviation
    '''
    def __init__(self,cnn_rand=True,STATIC=False,ExternalEmbeddingModel=None,n_symbols=None,wordmap=None,vocabsize=None):
        # Model hyperparameters
        self.embedding_dim=50##
        self.filter_sizes = (3, 8)
        self.num_filters = 10
        self.hidden_dims=100
        self.dropout_prob=(0.5,0.8)
        self.loss='categorical_crossentropy'
        self.optimizer= 'rmsprop'
        self.l1_reg=0
        self.l2_reg=3 ##according to kim14
        self.std=0.05 ## standard deviation
        # Training Parameters
        self.set_training_paramters(batch_size=64,num_epochs=10)
        self.set_processing_parameters(sequence_length=500,vocab_size=vocabsize) ## changed to fit short text
        # Defining Model Layers
        if cnn_rand:
            ##Embedding Layer Randomly initialized
            embedding_layer=Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size+1)
            Classes = dh.read_labels()
            n_classes = len(Classes)

        else:
            ## Use pretrained model
            #n_symbols, wordmap = dh.get_word_map_num_symbols()
            self.set_etxrernal_embedding(ExternalEmbeddingModel)
            vecDic = dh.GetVecDicFromGensim(self.ExternalEmbeddingModel)
            Classes = dh.read_labels()
            n_classes = len(Classes)
            ## Define Embedding Layer
            embedding_weights = dh.GetEmbeddingWeights(embedding_dim=300, n_symbols=n_symbols, wordmap=wordmap,
                                                       vecDic=vecDic)
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=n_symbols, trainable=STATIC)
            embedding_layer.build((None,))  # if you don't do this, the next step won't work
            embedding_layer.set_weights([embedding_weights])

        Sequence_in = Input(shape=(self.sequence_length,), dtype='int32')
        embedding_seq = embedding_layer(Sequence_in)
        x = Dropout(self.dropout_prob[0])(embedding_seq)
        ## define Core Convultional Layers
        conv_blocks = []
        for sz in self.filter_sizes:
            conv = Convolution1D(filters=self.num_filters,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(x)
            conv =  MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        x = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        x = Dropout(self.dropout_prob[1])(x)
        x = Dense(self.hidden_dims, activation="relu",kernel_initializer=RandomNormal(stddev=self.std),
                         kernel_regularizer=L1L2(l1=self.l1_reg,l2=self.l2_reg))(x)
        preds = Dense(n_classes, activation='softmax')(x)
        ## return graph model
        model = Model(Sequence_in, preds)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.model=model


class clstm(General): # inherits general
    '''
    CLSTM like based on Zhu 16
    paper link: https://arxiv.org/pdf/1511.08630.pdf
    '''
    def __init__(self,clstm_rand=True,STATIC=False,ExternalEmbeddingModel=None,EmbeddingType=None,n_symbols=None,wordmap=None):
        # Model hyperparameters
        self.embedding_dim=300##
        #self.filter_sizes = (3, 8)
        self.num_filters = 10
        self.hidden_dims=100
        self.dropout_prob=(0.5,0.8)
        self.loss='categorical_crossentropy'
        self.optimizer= 'rmsprop'
        self.l1_reg=0
        self.l2_reg=3 ##according to kim14
        self.std=0.05 ## standard deviation
        self.kernel_size=3
        # Training Parameters
        self.set_training_paramters(batch_size=64,num_epochs=10)
        self.set_processing_parameters(sequence_length=30,vocab_size=50000) ## changed to fit short text
        # Defining Model Layers
        if clstm_rand:
            ##Embedding Layer Randomly initialized
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size)
            Classes = dh.read_labels()
            n_classes = len(Classes)

        else:
            ## Use pretrained model
            # n_symbols, wordmap = dh.get_word_map_num_symbols()
            self.set_etxrernal_embedding(ExternalEmbeddingModel, ModelType=EmbeddingType)
            print(self.EmbeddingType)
            if self.EmbeddingType == "skipgram" or "CBOW":
                vecDic = dh.GetVecDicFromGensim(self.ExternalEmbeddingModel)
            elif self.EmbeddingType == "fastText":
                vecDic = dh.load_fasttext(self.ExternalEmbeddingModel)
            Classes = dh.read_labels()
            n_classes = len(Classes)
            ## Define Embedding Layer
            embedding_weights = dh.GetEmbeddingWeights(embedding_dim=300, n_symbols=n_symbols, wordmap=wordmap,
                                                       vecDic=vecDic)
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=n_symbols, trainable=STATIC)
            embedding_layer.build((None,))  # if you don't do this, the next step won't work
            embedding_layer.set_weights([embedding_weights])

        Sequence_in = Input(shape=(self.sequence_length,), dtype='int32')
        embedding_seq = embedding_layer(Sequence_in)
        x = Dropout(self.dropout_prob[0])(embedding_seq)
        ## define Core Convultional Layers

        conv = Convolution1D(filters=self.num_filters,
                                 kernel_size=self.kernel_size,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(x)
        conv = MaxPooling1D(pool_size=2)(conv)
        x = Dropout(self.dropout_prob[1])(conv)
        ## Till this point CNN model, Now change to LSTM
        x = LSTM(self.hidden_dims, kernel_initializer=RandomNormal(stddev=self.std),
                 kernel_regularizer=L1L2(l1=self.l1_reg, l2=self.l2_reg),return_sequences=False)(x)
        preds = Dense(n_classes, activation='softmax')(x)
        ## return graph model
        model = Model(Sequence_in, preds)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.model = model

class BasicLSTM(General): ## inherits General
    '''
    LSTM Our implementation
    '''
    def __init__(self,lstm_rand=True,STATIC=False,ExternalEmbeddingModel=None,EmbeddingType=None,n_symbols=None,wordmap=None):
        self.embedding_dim = 300
        self.hidden_dims = 100
        self.dropout_prob = (0.5, 0.8)
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'rmsprop'
        self.l1_reg = 0
        self.l2_reg = 3  ##according to kim14
        self.std = 0.05  ## standard deviation
        # Training Parameters
        self.set_training_paramters(batch_size=64, num_epochs=10)
        self.set_processing_parameters(sequence_length=30, vocab_size=50000)  ## changed to fit short text
        # Defining Model Layers        if clstm_rand:
        ##Embedding Layer Randomly initialized
        if lstm_rand:
            ##Embedding Layer Randomly initialized
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size)
            Classes = dh.read_labels()
            n_classes = len(Classes)

        else:
            ## Use pretrained model
            # n_symbols, wordmap = dh.get_word_map_num_symbols()
            self.set_etxrernal_embedding(ExternalEmbeddingModel, ModelType=EmbeddingType)
            if self.EmbeddingType == "skipgram" or self.EmbeddingType == "CBOW":
                vecDic = dh.GetVecDicFromGensim(self.ExternalEmbeddingModel)
            elif self.EmbeddingType == "fastText":
                vecDic = dh.load_fasttext(self.ExternalEmbeddingModel)
            Classes = dh.read_labels()
            n_classes = len(Classes)
            ## Define Embedding Layer
            embedding_weights = dh.GetEmbeddingWeights(embedding_dim=300, n_symbols=n_symbols, wordmap=wordmap,
                                                       vecDic=vecDic)
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=n_symbols, trainable=STATIC)
            embedding_layer.build((None,))  # if you don't do this, the next step won't work
            embedding_layer.set_weights([embedding_weights])

        Sequence_in = Input(shape=(self.sequence_length,), dtype='int32')
        embedding_seq = embedding_layer(Sequence_in)
        x = Dropout(self.dropout_prob[0])(embedding_seq)
        x = LSTM(self.hidden_dims, kernel_initializer=RandomNormal(stddev=self.std),
                 kernel_regularizer=L1L2(l1=self.l1_reg, l2=self.l2_reg), return_sequences=False)(x)
        preds = Dense(n_classes, activation='softmax')(x)
        ## return graph model
        model = Model(Sequence_in, preds)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.model = model

##

class BasicBiLSTM(General): ## inherits General
    '''
    BiLSTM Our implementation
    '''
    def __init__(self,bilstm_rand=True,STATIC=False,ExternalEmbeddingModel=None,EmbeddingType=None,n_symbols=None,wordmap=None):
        self.embedding_dim = 300
        self.hidden_dims = 100
        self.dropout_prob = (0.5, 0.8)
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'rmsprop'
        self.l1_reg = 0
        self.l2_reg = 3  ##according to kim14
        self.std = 0.05  ## standard deviation
        # Training Parameters
        self.set_training_paramters(batch_size=64, num_epochs=2)
        self.set_processing_parameters(sequence_length=30, vocab_size=50000)  ## changed to fit short text
        # Defining Model Layers        if clstm_rand:
        ##Embedding Layer Randomly initialized
        if bilstm_rand:
            ##Embedding Layer Randomly initialized
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size)
            Classes = dh.read_labels()
            n_classes = len(Classes)

        else:
            ## Use pretrained model
            # n_symbols, wordmap = dh.get_word_map_num_symbols()
            self.set_etxrernal_embedding(ExternalEmbeddingModel, ModelType=EmbeddingType)
            if self.EmbeddingType == "skipgram" or self.EmbeddingType == "CBOW":
                vecDic = dh.GetVecDicFromGensim(self.ExternalEmbeddingModel)
            elif self.EmbeddingType == "fastText":
                vecDic = dh.load_fasttext(self.ExternalEmbeddingModel)
            Classes = dh.read_labels()
            n_classes = len(Classes)
            ## Define Embedding Layer
            embedding_weights = dh.GetEmbeddingWeights(embedding_dim=300, n_symbols=n_symbols, wordmap=wordmap,
                                                       vecDic=vecDic)
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=n_symbols, trainable=STATIC)
            embedding_layer.build((None,))  # if you don't do this, the next step won't work
            embedding_layer.set_weights([embedding_weights])

        Sequence_in = Input(shape=(self.sequence_length,), dtype='int32')
        embedding_seq = embedding_layer(Sequence_in)
        x = Dropout(self.dropout_prob[0])(embedding_seq)
        x = Bidirectional(LSTM(self.hidden_dims, kernel_initializer=RandomNormal(stddev=self.std),
                 kernel_regularizer=L1L2(l1=self.l1_reg, l2=self.l2_reg), return_sequences=False))(x)
        preds = Dense(n_classes, activation='softmax')(x)
        ## return graph model
        model = Model(Sequence_in, preds)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.model = model

class BasicBiGRUs(General): ## inherits General
    '''
    BiLSTM Our implementation
    '''
    def __init__(self,BiGRU_rand=True,STATIC=False,ExternalEmbeddingModel=None,EmbeddingType=None,n_symbols=None,wordmap=None):
        self.embedding_dim = 300
        self.hidden_dims = 100
        self.dropout_prob = (0.5, 0.8)
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'rmsprop'
        self.l1_reg = 0
        self.l2_reg = 3  ##according to kim14
        self.std = 0.05  ## standard deviation
        # Training Parameters
        self.set_training_paramters(batch_size=64, num_epochs=10)
        self.set_processing_parameters(sequence_length=30, vocab_size=50000)  ## changed to fit short text
        # Defining Model Layers        if clstm_rand:
        ##Embedding Layer Randomly initialized
        if BiGRU_rand:
            ##Embedding Layer Randomly initialized
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size)
            Classes = dh.read_labels()
            n_classes = len(Classes)

        else:
            ## Use pretrained model
            # n_symbols, wordmap = dh.get_word_map_num_symbols()
            self.set_etxrernal_embedding(ExternalEmbeddingModel, ModelType=EmbeddingType)
            if self.EmbeddingType == "skipgram" or self.EmbeddingType == "CBOW":
                vecDic = dh.GetVecDicFromGensim(self.ExternalEmbeddingModel)
            elif self.EmbeddingType == "fastText":
                vecDic = dh.load_fasttext(self.ExternalEmbeddingModel)
            Classes = dh.read_labels()
            n_classes = len(Classes)
            ## Define Embedding Layer
            embedding_weights = dh.GetEmbeddingWeights(embedding_dim=300, n_symbols=n_symbols, wordmap=wordmap,
                                                       vecDic=vecDic)
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=n_symbols, trainable=STATIC)
            embedding_layer.build((None,))  # if you don't do this, the next step won't work
            embedding_layer.set_weights([embedding_weights])

        Sequence_in = Input(shape=(self.sequence_length,), dtype='int32')
        embedding_seq = embedding_layer(Sequence_in)
        x = Dropout(self.dropout_prob[0])(embedding_seq)
        x = Bidirectional(GRU(self.hidden_dims, kernel_initializer=RandomNormal(stddev=self.std),
                 kernel_regularizer=L1L2(l1=self.l1_reg, l2=self.l2_reg), return_sequences=False))(x)
        preds = Dense(n_classes, activation='softmax')(x)
        ## return graph model
        model = Model(Sequence_in, preds)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.model = model

####################################################

class AttentionBiGru(General):


     def __init__(self):
         self.dropout_prob = (0.5, 0.8)
         self.hidden_dims = 100
         self.std = 0.05
         self.l1_reg = 3
         self.l2_reg = 3
         self.loss = 'categorical_crossentropy'
         self.optimizer = 'rmsprop'
         self.sequence_length=30
         self.embedding_dim=300
         self.vocab_size=50000
         self.num_epochs=10
         self.batch_size=64

         ## Define the BiGRU model

         SeqIn= Input(shape=(self.sequence_length,),dtype='int32')
         embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size)(SeqIn)
         M1 = Dropout(self.dropout_prob[0])(embedding_layer)
         activations = Bidirectional(GRU(self.hidden_dims, kernel_initializer=RandomNormal(stddev=self.std),
                 kernel_regularizer=L1L2(l1=self.l1_reg, l2=self.l2_reg), return_sequences=True))(M1)

         ## Timedistributed dense for each activation

         attention = TimeDistributed(Dense(1,activation='tanh'))(activations)
         attention = Flatten()(attention)
         attention = Activation('softmax')(attention)
         attention = RepeatVector(int(2*self.hidden_dims))(attention)
         attention = Permute([2, 1])(attention)

         # apply the attention

         sent_representation = merge([activations, attention], mode='mul')
         sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

         Classes = dh.read_labels()
         n_classes = len(Classes)
         preds= Dense(n_classes, activation='softmax')(sent_representation)

         model = Model(input=SeqIn, output=preds)
         model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[])

         self.model = model
#####################################################

class AttentionBiLSTM(General):
    def __init__(self,att_rand=True,ExternalEmbeddingModel=None,EmbeddingType=None,n_symbols=None,wordmap=None,STATIC=True):
        self.dropout_prob = (0.36, 0.36)
        self.hidden_dims = 100
        self.std = 0.05
        self.l1_reg = 3
        self.l2_reg = 3
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'rmsprop'
        self.sequence_length = 30
        self.embedding_dim = 300
        self.vocab_size = 50000
        self.num_epochs = 5
        self.batch_size = 64

        ## Define the BiGRU model


        if att_rand:
            ##Embedding Layer Randomly initialized
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size)
        else:
            ## Use pretrained model
            # n_symbols, wordmap = dh.get_word_map_num_symbols()
            self.set_etxrernal_embedding(ExternalEmbeddingModel, ModelType=EmbeddingType)
            if self.EmbeddingType == "skipgram" or self.EmbeddingType == "CBOW":
                print('using '+self.EmbeddingType)
                vecDic = dh.GetVecDicFromGensim(self.ExternalEmbeddingModel)
            elif self.EmbeddingType == "fastText":
                print('using ' + self.EmbeddingType)
                vecDic = dh.load_fasttext(self.ExternalEmbeddingModel)
            Classes = dh.read_labels()
            n_classes = len(Classes)
            ## Define Embedding Layer
            embedding_weights = dh.GetEmbeddingWeights(embedding_dim=300, n_symbols=n_symbols, wordmap=wordmap,
                                                       vecDic=vecDic)
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=n_symbols, trainable=STATIC)
            embedding_layer.build((None,))  # if you don't do this, the next step won't work
            embedding_layer.set_weights([embedding_weights])

        ###################################################
        SeqIn = Input(shape=(self.sequence_length,), dtype='int32')
        embedding_seq = embedding_layer(SeqIn)
        M1 = Dropout(self.dropout_prob[0])(embedding_seq)
        #M1 = Activation('tanh')(embedding_seq)
        activations = Bidirectional(GRU(self.hidden_dims, kernel_initializer=RandomNormal(stddev=self.std),
                                        kernel_regularizer=L1L2(l1=self.l1_reg, l2=self.l2_reg),
                                        return_sequences=True))(M1)

        ## Timedistributed dense for each activation

        attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(2*(self.hidden_dims))(attention)
        attention = Permute([2, 1])(attention)

        # apply the attention

        sent_representation = merge([activations, attention], mode='mul')
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

        Classes = dh.read_labels()
        n_classes = len(Classes)
        preds = Dense(n_classes, activation='softmax')(sent_representation)

        model = Model(input=SeqIn, output=preds)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        self.model = model

