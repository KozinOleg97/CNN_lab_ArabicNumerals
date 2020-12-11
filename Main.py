import PySimpleGUI as sg
import numpy as np

from Running import *

listTrain = np.zeros((6600, 93, 13, 1))
listTest = np.zeros((2200, 93, 13, 1))


def readFile():
    trainData = 'D:\PytCharmProjects\CNNArabicNumerals\Datas\Train_Arabic_Digit.txt'
    testData = 'D:\PytCharmProjects\CNNArabicNumerals\Datas\Test_Arabic_Digit.txt'
    try:
        global listTrain
        listTrain = read(trainData, 6600)
        global listTest
        listTest = read(testData, 2200)
    except IOError:
        print("No file")
        return False

    return True


def read(strDatas, zn):
    list = np.zeros((zn, 93, 13, 1))
    with open(strDatas, ) as str:
        # iClass = 0
        iBlocks = -1
        iStr = 0
        for line in str:
            s = line.splitlines()
            if len(s[0]) > 15:
                strData = s[0].split(" ")
                arr = np.array(strData, dtype=np.float32)
                for i in range(0, 13):
                    # list[iClass][iBlocks][iStr][i] = arr[i]
                    list[iBlocks][iStr][i] = arr[i]
                iStr += 1
            else:
                # if iBlocks < zn:
                iBlocks += 1
                iStr = 0
                # else:
                #     iClass += 1
                #     iBlocks = 0
                #     iStr = 0
    return list


def main():
    #Running.train(listTrain, listTest, 1)
    #Running.test(listTest)
    history = None
    num_epochs = 1
    layout = [
        [sg.Button("Обучить"), sg.Button("Распознать")],
        [sg.Text("Эпохи"), sg.Input(key='-INPUT-', default_text="1", size=(7, 1), justification='center')],
        [sg.Output(size=(88, 20), key='out')]
    ]
    window = sg.Window('CNN', layout)
    while True:
        event, values = window.read()
        if event == "Обучить" or event == "Распознать":
            try:
                num_epochs = int(values['-INPUT-'])
            except ValueError:
                print("Ошибка значения, выполняется для:" + str(num_epochs))
        if event == "Обучить":
            try:
                Running.train(listTrain, listTest, num_epochs)
            except Exception as inst:
                print(inst)
        elif event == "Распознать":
            Running.test(listTest)
        if event in (sg.WIN_CLOSED, 'Quit'):
            break
    window.close()


if __name__ == '__main__':
    if readFile():
        main()
