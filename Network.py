import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Conv2D — сверточный слой
# Activation — слой активации
# MaxPooling2D -слой дискретизации на основе выборки.
#   Цель состоит в том, чтобы уменьшить выборку входного представления (изображение, выходную матрицу скрытого слоя и т. Д.),
#   Уменьшив его размерность и сделав допущения относительно предположений о свойствах, содержащихся в выбранных субрегионах.
#   Работает на основе выбора максимального значения из подвыборки размером pool_size.
# Dropout — слой прореживания для решения проблемы переобучения сети
# Flatten — преобразование в одномерный вектор
# Dense — полносвязный слой


class Running:
    @classmethod
    def train(cls, listTrain, listTest, num_epochs):
        train_answers = np.zeros((6600, 10))
        test_answers = np.zeros((2200, 10))
        train_answers = cls.form_output(train_answers, 660)
        test_answers = cls.form_output(test_answers, 220)

        print("Обучение...")
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(93, 13, 1)))  # Входной блок 93 строки, 13 столбцов, 1 элемент
        # #Операция максимальной подвыборки
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # слой выравнивания
        model.add(Dropout(0.25))
        # одно измерение
        model.add(Flatten())
        # полносвязный слой
        model.add(Dense(128, activation='relu'))
        # слой выравнивания
        model.add(Dropout(0.5))
        # полносвязный слой
        model.add(Dense(10, activation='softmax'))

        # блок по 93, 13 в строке, 1 значение
        # model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(93, 13, 1)))
        # model.add(MaxPooling2D())
        # model.add(Conv2D(128, kernel_size=3, activation='relu'))
        # model.add(Flatten())  # – слой, преобразующий 2D-данные в 1D-данные.
        # model.add(Dense(10, activation='softmax'))
        #############################################

        # optimizer = 'adam'(Адам: метод стохастической оптимизации).Функция потерь: loss = 'categorical_crossentropy'
        # категориальная перекрестная энтропия(categorical crossentropy CCE)
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

        history = model.fit(listTrain, train_answers, epochs=num_epochs, validation_data=(listTest, test_answers))

        print("История:")
        print(history.history)
        model.save('Model')
        cls.draw_graph(history)

    # формируем структуру выхода (10 классов (по датасету))
    @classmethod
    def form_output(cls, output_list, data_len):
        for j in range(0, 10):
            for i in range(0, data_len):
                val = np.zeros(10)
                val[j] = 1
                output_list[j * data_len + i] = val
        return output_list

    @classmethod
    def draw_graph(cls, history):
        plt.subplot(2, 1, 1)
        plt.plot(history.history['accuracy'])
        plt.title('Точность при обучении')
        plt.ylabel('Точность')
        plt.xlabel('Эпохи')
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.title('Потери при обучении')
        plt.ylabel('Потеря')
        plt.xlabel('Эпохи')
        plt.show()

    @classmethod
    def test(cls, listTest):
        print("Тест")
        model = tf.keras.models.load_model('Model')
        test_answers = np.zeros((2200, 10))
        test_answers = cls.form_output(test_answers, 220)
        score = model.evaluate(listTest, test_answers, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        res = model.predict(listTest)
        c = 1
        while c != 0:
            print("Enter index, 0 - Exit")
            c = int(input())
            elem = res[c]
            max = 0
            index = -1
            for i in range(10):
                if max < elem[i]:
                    max = elem[i]
                    index = i
            print(index + 1)
            max = 0
            index = 0
