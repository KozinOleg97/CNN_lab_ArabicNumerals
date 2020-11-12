import PySimpleGUI as sg
import splitfolders
import numpy as np

from Running import *

listTrain = []
listTest = []


def readFile():
    trainData = 'D:\PytCharmProjects\CNNArabicNumerals\Datas\Train_Arabic_Digit.txt'
    testData = 'D:\PytCharmProjects\CNNArabicNumerals\Datas\Test_Arabic_Digit.txt'
    try:
        global listTrain
        listTrain = read(trainData, 661)
        global listTest
        listTest = read(testData, 221)
    except IOError:
        print("No file")
        return False

    return True


def read(strDatas, zn):
    list = []
    with open(strDatas, ) as str:
        count = 0
        iClass = 0
        iBlocks = -1
        list.append([])
        for line in str:
            s = line.splitlines()
            if len(s[0]) > 15:
                strData = s[0].split(" ")
                arr = np.array(strData, dtype=np.float32)
                list[iClass][iBlocks].append(arr)
                #list[iBlocks].append(arr)
            else:
                # можно удалить
                #iBlocks += 1
                #if iBlocks < 6599:
                #    list.append([])
                count += 1
                if count < zn:
                    iBlocks += 1
                    list[iClass].append([])
            if count == zn:
                count = 1
                iClass += 1
                iBlocks = 0
                list.append([[]])
    return list


def main():
    Running.train(listTrain, 1)
    # num_epochs = 1
    # layout = [
    #     [sg.Button("Обучить"), sg.Button("Распознать")],
    #     [sg.Text("Эпохи"), sg.Input(key='-INPUT-', default_text="1", size=(7, 1), justification='center')],
    #     [sg.Output(size=(88, 20), key='out')]
    # ]
    # window = sg.Window('CNN', layout)
    # while True:
    #     event, values = window.read()
    #     if event == "Обучить" or event == "Распознать":
    #         try:
    #             num_epochs = int(values['-INPUT-'])
    #         except ValueError:
    #             print("Ошибка значения, выполняется для:" + str(num_epochs))
    #     if event == "Обучить":
    #         try:
    #             Running.train(listTrain, num_epochs)
    #         except Exception as inst:
    #             print(inst)
    #     elif event == "Распознать":
    #         Running.test(listTest, num_epochs)
    #     if event in (sg.WIN_CLOSED, 'Quit'):
    #         break
    # window.close()


if __name__ == '__main__':
    if readFile():
        main()
