# coding=utf-8

from keras import Input, Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten,Embedding,Dropout,Bidirectional,Activation,LSTM,concatenate
from keras.utils.vis_utils import plot_model
import os

os.environ["PATH"] += os.pathsep +'C:/Program Files (x86)/Graphviz2.38/bin/'


class TextCNN(object):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=3,
                 last_activation='sigmoid',filters=2):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.filters = filters

    def get_model(self):
        
        seq = Input(shape=[self.maxlen],name='x_seq')
        seq = Input((self.maxlen,))
        #Embedding layers
        emb = Embedding(self.max_features,self.embedding_dims)(seq)
        convs = []
        filter_sizes = [2,3,4]
        for fsz in filter_sizes:
            conv1 = Conv1D(self.filters,kernel_size=fsz,activation='tanh')(emb)
            pool1 = MaxPooling1D(self.maxlen-fsz+1)(conv1)
            pool1 = Flatten()(pool1)
            convs.append(pool1)
        merge0 = concatenate(convs,axis=1)
        out = Dropout(0.5)(merge0)
        encoded_text = Bidirectional(LSTM(256,return_sequences = False),merge_mode='sum')(emb)       
        merge = concatenate([out,encoded_text],axis=1)
        output = Dense(32,activation='relu')(merge)
        output = Dense(3,activation='softmax')(output)
        model = Model([seq],output)
        
#        model.summary()
#        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#        plot_model(model,show_shapes = True,to_file = 'model.png')	
        
        return model
        



