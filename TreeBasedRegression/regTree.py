#-*- coding:utf-8-*-
'回归树'
__author__ = 'dell'

class treeNode():
    '''
    定义树节点
    '''
    def __init__(self, feat, val, right, left):
        featureToSpliton = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left


import numpy as np

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0],:][0]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0,mat1

def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    '''
    创建树
    :param dataSet:
    :param leafType:
     建立叶节点的函数
    :param errType:
    计算错误率
    :param ops:
    存储创建树的所需要的其他参数的元组
    :return:
    '''
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

if __name__ == "__main__":
    testMat = np.mat(np.eye(4))
    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)

