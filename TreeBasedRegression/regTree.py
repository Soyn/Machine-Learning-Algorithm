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

def regLeaf(dataSet):
    '''
    定义回归树的叶子节点，返回目标变量的均值
    :param dataSet:
    :return:
    '''
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    '''

    :param dataSet:
    :return:
    '''
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

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


def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    '''
    选择最佳的二分处理数据的方法
    :param dataSet:
    :param leafType:
    :param errType:
    :param ops:
    :return:
    '''
    import  math
    tols = ops[0]#允许的误差下降值
    tolN = ops[1]#最小的切分样本数
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:#数据集中的值相等时
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tols:
        return None, leafType
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN ):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def isTree(obj):
    '''
    检测obj是不是树
    :param obj:
    :return:
    '''
    return  (type(obj).__name__ == 'dict')

def getMean(tree):
    '''
    从根节点开始遍历至叶子节点
    :param tree:
    :return:
    返回叶子节点的平均值
    '''
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    '''
    树剪枝
    :param tree:
    :param testData:
    用于剪枝的数据
    :return:
    '''
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['left']) or isTree(tree['right'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):#到达叶子节点后
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'],2)) + np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print "mergeing"
            return treeMean
        else:
            return tree
    else:
        return tree

if __name__ == "__main__":
    myDat  = loadDataSet('ex2.txt')
    myMat = np.mat(myDat)
    myTree = createTree(myMat, ops = (0, 1))
    myDatTest = loadDataSet('ex2test.txt')
    myMat2Test = np.mat(myDatTest)
    print prune(myTree, myMat2Test)


