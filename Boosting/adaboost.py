#-*- coding: utf-8 -*-
__author__ = 'dell'

'自适应Boost(adaboost)'

import numpy as np
import math

def loadSimpleData():
    dataMat = np.matrix(
        [[1.0, 2.1],
        [2.0, 1.1 ],
        [1.3, 1.0],
        [1.0, 1.0],
        [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return dataMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    和厥值比较分类
    :param dataMatrix:
    :param dimen:
    :param threshVal:
    :param threshIneq:
    :return:
    '''
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    '''
    获得最佳的单层决策树
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    '''
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}#存储给定权重D的最佳单层决策树的信息
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf

    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[: , i].max()
        #计算出步长
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr

                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

def adaBoostTrainDs(dataArr, classLabels, numIt = 40):
    '''
    利用单层决策树训练adaBoost算法
    :param dataArr:
    :param classLabels:
    :param numIt:
    :return:
    '''
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #print 'D:  ', D.T
        alpha = float(0.5 * math.log((1.0 - error) / max(error, math.e - 16)))
        bestStump['alpha'] = alpha

        weakClassArr.append(bestStump)
        #print "classEst:  ", classEst.T
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        aggClassEst += alpha * classEst
        #print "aggClassEst:  ", aggClassEst.T
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m

        print "total error:  ", errorRate, '\n'
        if errorRate == 0.0:
            break
    return weakClassArr

def adaClassify(dataToClass, classifierArr):
    '''
    基于adaBoost的分类函数
    :param dataToClass:
    待分类的实例
    :param classifierArr:

    :return:
    '''
    dataMatrix = np.mat(dataToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return np.sign(aggClassEst)

def loadDataSet(filename):
    '''
    从文件中加载数据
    :param filename:
    :return:
    '''
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


if __name__ == "__main__":
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArrary = adaBoostTrainDs(dataArr, labelArr, 10)
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClassify(testArr, classifierArrary)
    errArr = np.mat(np.ones((67, 1)))
    print errArr[prediction10 != np.mat(testLabelArr).T].sum()