import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
import pandas as pd

class DataGenerator(Sequence):
    def __init__(self,
                 phase,
                 max_audio_length, 
                 sample_rate=16000, 
                 csv_file=None, 
                 n_fft=510, hop_length=160, 
                 win_length=400, 
                 X_col='audio_path', y_col='label',
                 batch_size=32,
                 shuffle=True):
        self.batch_size = batch_size
        self.phase = phase
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_audio_length = max_audio_length 

        print('Loading data...')
        
        self.df = pd.read_csv(csv_file, delimiter=';')
        self.X_col = X_col
        self.y_col = y_col
       
        self.shuffle = shuffle
       
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            print('Shuffle')
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.df) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temps = [self.img_indexes[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temps)
        return X, y
 
    def __data_generation(self, list_IDs_temps):
        X = np.empty((self.batch_size, *self.dim))
        y = []
        for i, ID in enumerate(list_IDs_temps):
            X[i,] = self.img_paths[ID]
            X = (X/255).astype('float32')
            y.append(self.labels[ID])
        X = X[:,:,:, np.newaxis]
        return X, keras.utils.to_categorical(y, num_classes=10)


n_classes = 10
input_shape = (28, 28)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28 , 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

train_generator = DataGenerator(x_train, y_train, batch_size = 32, dim = input_shape,
 n_classes=10, shuffle=True)
val_generator = DataGenerator(x_test, y_test, batch_size=32, dim = input_shape, 
n_classes= n_classes, shuffle=True)

model.fit_generator(
 train_generator,
 steps_per_epoch=len(train_generator),
 epochs=10,
 validation_data=val_generator,
 validation_steps=len(val_generator))
