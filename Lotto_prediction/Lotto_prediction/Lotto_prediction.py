import pandas as pd
import numpy as np
import threading
import time
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

class PredictionLOTTO :
    def __init__(self, windowLength) :
        self.window_length = windowLength

    def trainData(self) :
        self.df = pd.read_csv('lotto_new.csv')
        self.scaler = StandardScaler().fit(self.df.values)
        self.transformed_dataset = self.scaler.transform(self.df.values)
        self.transformed_df = pd.DataFrame(data=self.transformed_dataset, index=self.df.index)
        self.number_of_rows= self.df.values.shape[0] #전체 게임 횟수
        self.number_of_features = self.df.values.shape[1] #공 갯수
        self.train = np.empty([self.number_of_rows - self.window_length, self.window_length, self.number_of_features], dtype=float)
        self.label = np.empty([self.number_of_rows - self.window_length, self.number_of_features], dtype=float)
        for i in range(0, self.number_of_rows - self.window_length):
            self.train[i]=self.transformed_df.iloc[i:i+self.window_length, 0: self.number_of_features]
            self.label[i]=self.transformed_df.iloc[i+self.window_length: i+self.window_length+1, 0: self.number_of_features]

        self.batch_size = 25
        self.model = Sequential()
        self.model.add(LSTM(32, input_shape=(self.window_length, self.number_of_features), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(32, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.number_of_features))
        self.model.compile(loss='mse', optimizer='rmsprop')
        self.model.fit(self.train, self.label, batch_size=931, epochs=8600)#86400

    def prediction(self) :
        self.to_predict = self.df.iloc[len(self.df) - self.window_length : len(self.df)]
        self.scaled_to_predict = self.scaler.transform(self.to_predict)
        self.scaled_predicted_output_1 = self.model.predict(np.array([self.scaled_to_predict]))
        return self.scaler.inverse_transform(self.scaled_predicted_output_1).astype(int)[0]

predicList = []
lock = threading.Lock()

preLoto = PredictionLOTTO(10)
start = time.time()
for i in range(0, 5) :
    preLoto.trainData()
    predicList.append(str(preLoto.prediction()))

print('Time : ', time.time() - start)
print(predicList)

f = open('로또예측.txt', 'w')
for i in range(0, 5):
    data = predicList[i]
    f.write(data)
f.close()