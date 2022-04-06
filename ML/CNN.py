from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 读取一个路径下所有的.dat文件，默认路径设为..\\DATA\\xxxx
os.chdir(os.path.realpath(__file__) + "\\..")
train_path_list = os.listdir("DATA\\pre3")
os.chdir("DATA\\pre3")


data = pd.read_csv(r'20210725Apre3.csv')

print(data.shape)

list = range(1, 75)
randomlist = random.sample(list, 15)

print(randomlist)
data1 = data[data['th'].isin(randomlist)]
data2 = data[~data['th'].isin(randomlist)]

X_train = data1.iloc[:, 5:905]
Y_train = data1.iloc[:, data.columns == "bin_type"]

X_test = data2.iloc[:, 5:905]
Y_test = data2.iloc[:, data.columns == "bin_type"]

standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train)

X_train = standard_scaler.transform(X_train)
X_test = standard_scaler.transform(X_test)

X_train = X_train.reshape((X_train.shape[0], 30, 30, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 30, 30, 1)).astype('float32')

print(X_train.shape)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

print(Y_test.shape)

num_classes = Y_test.shape[1]

model = Sequential()

# 一层卷积层 包含64个卷积核 大小5*5
model.add(Conv2D(64, (5, 5), input_shape=(30, 30, 1), activation='relu', data_format="channels_last", padding="same"))
model.add(Conv2D(64, (5, 5), input_shape=(30, 30, 1), activation='relu', data_format="channels_last", padding="same"))

# 一个最大池化层 池化大小为2*2
model.add(MaxPooling2D(pool_size=(2, 2)))

# 一个卷积层包含128个卷积核 3*3
model.add(Conv2D(128, (3, 3), activation='relu', data_format="channels_last", padding="same"))
model.add(Conv2D(128, (3, 3), activation='relu', data_format="channels_last", padding="same"))

# 一个池化层
model.add(MaxPooling2D(pool_size=(2, 2)))

# 遗忘层
model.add(Dropout(0.25))

# 压平层
model.add(Flatten())

# 全连接
model.add(Dense(128, activation='relu'))
# 分类
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=256, verbose=2)


# Final evaluation of the model
scores = model.evaluate(X_test,Y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


model.save('weights.model')
model.save_weights("model.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)