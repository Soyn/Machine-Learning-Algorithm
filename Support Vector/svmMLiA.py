#-*- coding: utf-8 -*-
__author__ = 'dell'

'SVM算法的简化版本SMO'

import random as nd
import numpy as np
def loadDataSet(filename):
    '''
    从文件中读取数据
    :param filename:
    :return:
    以列表形式返回数据矩阵，和类别
    '''
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    '''

    :param
    i: alpha的索引
    :param m:
    alpha的数量
    :return:
    '''
    j = 1
    while(j == i):
        j = int(nd.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    SMO算法
    :param dataMatIn:
     数据集
    :param classLabels:
     类别标签
    :param C:
     C为常量
    :param toler:
     toler为容错
    :param maxIter:
     最大迭代次数
    :return:
    '''
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()#转置矩阵
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0#没有任何alphas改变的情况下遍历数据集的次数

    while(iter < maxIter):
        alphaPairsChanged = 0#记录alphas是否优化了
        for i in range(m):

            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b#预测类
            Ei = fXi -float(labelMat[i])#计算误差

            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C) ) or \
                    ((labelMat[i] * Ei > toler) and (alphas[i] > 0 )): #当alpha可以优化时

                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat). T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])

                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #L和H来用于将label[i]限制在0~C
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C,  alphas[j] + alphas[i])

                if L == H:print "L == H" ;continue

                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T

                if eta >= 0:   print "eta>=0";continue
                alphas[j] -= labelMat[j] * (Ei - Ej) /eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                if(np.abs(alphas[j] - alphaJold) < 0.00001):

                    print "j not moving enough";continue

                alphas[i]  += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) *\
                              dataMatrix[i, :] * dataMatrix[i, :].T -\
                     labelMat[j] * (alphas[j] - alphaJold)*\
                    dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) *\
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if 0 < alphas[i] and (C > alphas[j]):
                    b = b1
                elif 0 < alphas[j] and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print "iter: %d i: %d, pairs changed %d " % (iter, i, alphaPairsChanged)

        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" %iter
    return b,alphas



if __name__ == "__main__":
    dataArr, labelArr = loadDataSet('testSet.txt')
    b,alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print labelArr
