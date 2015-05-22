#-*- coding: utf-8 -*-
__author__ = 'dell'

'自适应Boost(adaboost)'

import numpy as np

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


