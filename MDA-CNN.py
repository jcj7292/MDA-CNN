import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Input, Add, Conv2D, Flatten

# Continuous function with linear correlation
def y_L_function(x):
    A, B, C = 0.5, 10, -5
    return A*(6*x-2)**2*np.sin(12*x-4) + B*(x-0.5) + C
def y_H_function(x):
    return (6*x-2)**2*np.sin(12*x-4) 
N_L, N_H = 21, 4

x_L = np.linspace(0.0, 1.0, num=N_L)
x_H = np.linspace(0.0, 1.0, num=N_H)

y_train_H = y_H_function(x_H)

train_data = np.empty((len(x_H), len(x_L), 4))
train_data[:] = np.NaN
for i in range(len(x_H)):
    train_data[i,:,0] = x_L
    train_data[i,:,1] = y_L_function(x_L)
    train_data[i,:,2] = [x_H[i]]*len(x_L)
    train_data[i,:,3] = [y_L_function(x_H[i])]*len(x_L)


def cus_cost(y_prediction, y_real):
    return tf.reduce_mean(tf.math.square(y_prediction - y_real) * tf.constant([[7.0]]))

epochs = 5000
batch_size = 10
learning_rate = 0.001

InputLayer = Input(shape=(len(x_L), train_data.shape[2], 1))
ConvolutionalLayer = Conv2D(filters=64, kernel_size=[3, train_data.shape[2]])(InputLayer)
F_l = Dense(1, activation="linear")(Flatten()(ConvolutionalLayer))
Layer_nlin_1 = Dense(10, activation="tanh", kernel_regularizer=regularizers.l1(0.01))(Flatten()(ConvolutionalLayer))
Layer_nlin_2 = Dense(10, activation="tanh", kernel_regularizer=regularizers.l1(0.01))(Layer_nlin_1)
F_nl = Dense(1, activation="linear")(Layer_nlin_2)
OutputLayer_y_H = Add()([F_l, F_nl])

InputLayer_y_H = Input(shape=(1,))
model = Model(inputs=[InputLayer, InputLayer_y_H], outputs=[OutputLayer_y_H])
lossF = cus_cost(OutputLayer_y_H, InputLayer_y_H)
model.add_loss(lossF)
adamOptimizer = optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=adamOptimizer,metrics=['mse'])  
history_cache = model.fit([train_data, y_train_H],
                          verbose=1,
                          epochs=epochs,
                          batch_size=batch_size)
print('Final cost: {0:.4f}'.format(history_cache.history['loss'][-1]))


x_test = np.linspace(np.min(x_L), np.max(x_L), num=101)

test_data = np.empty((len(x_test), len(x_L), 4))
test_data[:] = np.NaN
for i in range(len(x_test)):
    test_data[i,:,0] = x_L
    test_data[i,:,1] = y_L_function(x_L)
    test_data[i,:,2] = [x_test[i]]*len(x_L)
    test_data[i,:,3] = [y_L_function(x_test[i])]*len(x_L)

y_pred = model.predict(list((test_data, x_test)))


plt.figure(figsize=(5, 3))
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel('y')
plt.ylabel('Q (y)')
plt.scatter(x_L, y_L_function(x_L),12,c='b',label='LF data')
plt.scatter(x_H, y_H_function(x_H),c='r',label='HF data')
plt.plot(x_test, y_L_function(x_test), '-.', label='LF model')
plt.plot(x_test, y_H_function(x_test), '-', label='HF model')
plt.plot(x_test, y_pred, '--', label='MF')
plt.legend(ncol=2, loc='upper left',prop={'size': 10},frameon=False)
plt.show()

plt.figure(figsize=(3, 3))
plt.plot(history_cache.history['loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

