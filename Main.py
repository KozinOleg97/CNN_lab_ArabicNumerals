import os

import numpy as np

from Network import *

PATH = os.path.abspath(os.getcwd())

listTrain = np.zeros((6600, 93, 13, 1))
listTest = np.zeros((2200, 93, 13, 1))


def readFile():
    trainData = '/home/alexander/Projects/CNN_lab_ArabicNumerals/Datas/Train_Arabic_Digit.txt'
    testData = '/home/alexander/Projects/CNN_lab_ArabicNumerals/Datas/Test_Arabic_Digit.txt'
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
    with open(strDatas, ) as str_list:
        # iClass = 0
        iBlocks = -1
        iStr = 0
        for line in str_list:
            s = line.splitlines()
            if len(s[0]) > 15:
                strData = s[0].split(" ")
                arr = np.array(strData, dtype=np.float32)
                for i in range(0, 13):
                    list[iBlocks][iStr][i] = arr[i]
                iStr += 1
            else:
                iBlocks += 1
                iStr = 0
    return list


def main():
    history = None
    num_epochs = 1

    while True:
        print("  1 - train, 2 - test, 0 - close")
        res = input()
        if res == '1':
            print("     Enter epoch number")
            num_epochs = int(input())
            try:
                Running.train(listTrain, listTest, num_epochs)
            except Exception as inst:
                print(inst)

        elif res == '2':
            Running.test(listTest)

        elif res == '0':
            break


if __name__ == '__main__':
    if readFile():
        main()
