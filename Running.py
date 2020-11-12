import tensorflow as tf

from CNN import *
import numpy as np
from tqdm import tqdm
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Activation, Flatten, Dropout, Dense, MaxPooling2D
from tensorflow_core.python.keras.optimizers import SGD
from tensorflow.python.keras.utils.data_utils import Sequence


class Running:
    @classmethod
    def train(cls, listTrain, num_epochs):
        print("Обучение...")
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=( 93, 13, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        model.summary()

        model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
        y = []
        for i in range(0, 10):
            value = tf.keras.utils.to_categorical(i, 10)
            y.append(value)
        model.fit(listTrain, y, epochs=num_epochs)

        # model = Sequential()
        # # первый скрытый слой
        # model.add(Conv2D(32, (13, 1), strides=(2, 2), input_shape=(93, 13, 1)))
        # model.add(AveragePooling2D((2, 1), strides=(2, 2)))
        # model.add(Activation('relu'))
        # # второй скрытый слой
        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(AveragePooling2D((2, 1), strides=(2, 2)))
        # model.add(Activation('relu'))
        # # третий скрытый слой
        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(AveragePooling2D((2, 1), strides=(2, 2)))
        # model.add(Activation('relu'))
        # # слой выравнивания
        # model.add(Flatten())
        # model.add(Dropout(rate=0.5))
        # # полносвязный слой
        # model.add(Dense(64))
        # model.add(Activation('relu'))
        # model.add(Dropout(rate=0.5))
        # # выходной слой
        # model.add(Dense(10))
        # model.add(Activation('softmax'))
        # model.summary()
        #
        # model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'])
        #
        # y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # for i in range(0, 10):
        #     y[i] = tf.keras.utils.to_categorical(i, 10)
        # model.fit(listTrain, y, epochs=num_epochs)

        # Оценка модели
        # model.evaluate_generator(generator=listTrain, steps=epochs)

        # m = 5000
        # num_classes = 10
        # lr = 0.01
        # beta1 = 0.95
        # beta2 = 0.99
        # img_dim = 28
        # img_depth = 1
        # f = 5
        # num_filt1 = 8
        # num_filt2 = 8
        # batch_size = 32
        #
        #
        # f1, f2, w3, w4 = (num_filt1, img_depth, f, f), (num_filt2, num_filt1, f, f), (128, 800), (10, 128)
        # f1 = CNN.initializeFilter(f1)
        # f2 = CNN.initializeFilter(f2)
        # w3 = CNN.initializeWeight(w3)
        # w4 = CNN.initializeWeight(w4)
        #
        # b1 = np.zeros((f1.shape[0], 1))
        # b2 = np.zeros((f2.shape[0], 1))
        # b3 = np.zeros((w3.shape[0], 1))
        # b4 = np.zeros((w4.shape[0], 1))
        #
        # params = [f1, f2, w3, w4, b1, b2, b3, b4]
        #
        # cost = []
        #
        # print("LR:" + str(lr) + ", Batch Size:" + str(batch_size))
        # train_data = np.array([listTrain[0], listTrain[1], listTrain[2], listTrain[3], listTrain[4]])
        #
        #
        # for epoch in range(num_epochs):
        #     np.random.shuffle(listTrain)
        #     batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]
        #
        #     t = tqdm(batches)
        #     for x, batch in enumerate(t):
        #         params, cost = CNN.adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
        #         t.set_description("Cost: %.2f" % (cost[-1]))

        # return cost

    @classmethod
    def test(cls, listTest, num_epochs):
        print("Тест...")
        # test_set.reset()
        # pred = model.predict_generator(test_set, steps=num_epochs, verbose=1)
