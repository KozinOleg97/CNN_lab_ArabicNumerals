import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Activation, Flatten, Dropout, Dense, MaxPooling2D
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Running:
    @classmethod
    def train(cls, listTrain, listTest, num_epochs):
        train_answers = np.zeros((6600, 10))
        test_answers = np.zeros((2200, 10))
        train_answers = cls.fill_answers(train_answers, 93, 13)
        test_answers = cls.fill_answers(test_answers, 93, 13)

        print("Обучение...")
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(93, 13, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        model.summary()

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(listTrain, train_answers, epochs=num_epochs, validation_data=(listTest, test_answers))

        model.save('Model')

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

    @classmethod
    def fill_answers(cls, answers_list, block_size, data_len):
        for i in range(0, data_len):
            value = int(i / block_size)
            value = tf.keras.utils.to_categorical(value, 10)
            answers_list[i] = value
        return answers_list

    @classmethod
    def test(cls, listTest):
        print("Тест...")
        model = tf.keras.models.load_model('Model')
        test_answers = np.zeros((2200, 10))
        test_answers = cls.fill_answers(test_answers, 93, 13)
        score = model.evaluate(listTest, test_answers, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])