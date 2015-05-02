#-*- coding: utf-8 -*-
__author__ = 'dell'

'''
	创建决策树
'''

from math import log
import operator

def createDataSet():
    dataSet = [[1,1 ,'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels



def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    #用字典存储标签：
    labelCounts = {}
    for featVec in dataSet:
        #读取dataSet中的标签：
        currentLabel = featVec[-1]
        #将标签计数：
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    #计算shannon Entopy
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    '''
    dataSet: 将要分割的数据
    axis:分割的数据
    value：将要返回的特点的值
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #将挑选出来的数据存入reducedFeatVec:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1 :])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1#最后一行用来存储标签
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        #将dataSet中的重复元素剔除存在featList中
        featList = [example[i] for example in dataSet]#创建一个存储feature的列表
        uniqueVals = set(featList)
        newEntropy = 0.0
        #计算每一个分组的熵：
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) /float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    #创建字典，字典的key是classList的值，key对应的值是其出现的频率：
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys() :
            classCount[vote] = 0
        classCount += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0] #返回出现频率最高的数据

def createTree(dataset, labels):
    '''
    创建决策树
    '''

    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):#当类一样时，递归结束
        return classList[0]
    if len(dataset[0]) == 1:
        #当找不到特性时，返回majority，即达到叶子节点时，递归结束
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel : {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat, value), subLabels)
    return myTree

if __name__ == '__main__':
    myDat, labels = createDataSet()
    myTree = createTree(myDat, labels)
    print myTree
    #print myDat
    #print chooseBestFeatureToSplit(myDat)

    '''
    print splitDataSet(myDat, 0, 1)
    #print calcShannonEnt(myDat)'''