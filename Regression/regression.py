#-*- coding:utf-8 -*-
__author__ = 'dell'

'利用回归分类'

import numpy as np
from math import *

def loadDataSet(filename):
    '''
    从文件中获得数据集
    :param filename:
    :return:
    '''
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat =[]
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    '''
    计算最佳的曲线
    :param xArr:
    :param yArr:
    :return:
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print "This martix is singular, cannot do inverse."
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def PlotBestFit():
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    yHat = xMat * ws
    print np.corrcoef(yHat.T, yMat)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


def lwlr(testPoint, xArr, yArr, k = 1.0):
    '''
    计算局部权重线性回归
    :param testPoint:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))#创建对角线矩阵
    for j in range(m):
        diffMat = testPoint -  xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k **2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def PlotLocalRegression():
    import  matplotlib.pyplot as plt
    xArr, yArr = loadDataSet('ex0.txt')
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = np.mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s = 2, c = 'red')
    plt.show()

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def ridgeRegres(xMat, yMat, lam = 0.2):
    '''
    利用岭回归处理特征值大于数据点时
    :param xMat:
    :param yMat:
    :param lam:
    :return:
    '''
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print "this matrix is singular, cannot do inverse."
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    import  math
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, math.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

def PlotRidgeRegres():
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

def regularize(xMat):
    inMat = xMat.copy()
    inMeans  = np.mean(inMat, 0)
    inVar = np.var(inMat, 0)
    inMat = (inMat - inMeans) / inVar
    return inMat
def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    import math
    '''
    前向逐步回归算法实现
    :param xArr:
    :param yArr:
    :param eps:
    每次迭代的步长
    :param numIt:
    迭代的总次数
    :return:
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = np.inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat

if __name__ == "__main__":
    xArr,yArr = loadDataSet('abalone.txt')
    print stageWise(xArr, yArr, 0.01, 200)



