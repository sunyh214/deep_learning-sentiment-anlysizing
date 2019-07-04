# coding=utf-8

from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from data_preprocess import get_data
from text_cnn import TextCNN
from text_birnn import TextBiRNN

max_features = 10000
maxlen = 200
batch_size = 128
embedding_dims = 256
epochs = 100

print('Loading data...')
n_symbols,embedding_weights,x_train,y_train,x_test,y_test = get_data()
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = TextCNN(maxlen, max_features, embedding_dims).get_model()
#model = TextBiRNN(maxlen, max_features, embedding_dims).get_model()
#model = TextRCNN(maxlen, max_features, embedding_dims).get_model()
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
#early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
#          callbacks=[early_stopping],
          validation_data=(x_test, y_test))

print('Test...')
result = model.predict(x_test)
