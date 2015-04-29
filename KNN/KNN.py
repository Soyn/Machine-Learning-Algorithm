#-*- coding: utf-8-*-
__author__ = 'dell'
'K-Nearest-Neighbors'

from numpy import *
import operator
import os
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    '''
    设置数据和标签
    :return:
    '''
    group = array([[1.0, 1.1], [1.0, 1.0],[0,0], [0, 0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX, dataSet, labels,k):
    '''

    :param
    inX:输入的数据
    :param
    dataset:已经存储的数据标准
    :param
    labels:数据的标签
    :param
    k:设立的与标准的偏移量
    :return:
    返回预测结果
    '''
    #计算inX与dataSet的距离：
    dataSetSize = dataSet.shape[0] #获取矩阵的行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #生成行数为dataSetSize,列数为1的矩阵，元素都为inX
    sqDiffMat = diffMat ** 2
    #矩阵的每一行相加构成一个元素：
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    #将算出的距离排序，默认升序,返回值为列表的索引：
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1),  reverse = True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    '''
    将文件装换为矩阵
    :param filename:
    :return:
    '''
    #获得文本的行数
    loveDictionary = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    #创建矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        #删除每行中的空白符（例如：'\n','\r','\t等'）
        line = line.strip()
        #将用'\t'隔开的元素组成list
        listFromLine = line.split('\t')
        #将前三列元素作为矩阵的每一行
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1].isdigit():
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(loveDictionary.get(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector

def autoNorm(dataSet):
    '''
    将数据放缩到同一个区间（0~1）
    :param dataSet:
    :return:
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    #创建矩阵：scaling
    normDataset = zeros(shape(dataSet))
    #获得矩阵的行数：
    m = dataSet.shape[0]
    #tile获得元素值都为minVals，行数为m，列数为1的矩阵：
    normDataset = dataSet - tile(minVals, (m, 1))
    normDataset = normDataset / tile(ranges, (m, 1))
    return normDataset, ranges, minVals


def datingClassTest():
    '''
    测试KNN算法的错误率
    :return:
    '''
    #取数据的前10%测试
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #获取矩阵的行数
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs : m, :], datingLabels[numTestVecs : m], 3)
        print "the classifier came back with: %d, the real answer is: %d"  % (classifierResult, datingLabels[i])
        if  classifierResult != datingLabels[i]:
            errorCount += 1.0
    print "the total error rate is %f" %(errorCount/float(numTestVecs))


def classifyPerson():
    resultList = ['not at all ',' in small doses','in large doses' ]
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream,])
    classifierResult = classify0( (inArr - minVals) / ranges, normMat, datingLabels, 3)
    print "You will probably like this person: % s" % resultList[classifierResult - 1]


def img2Vector(filename):
    '''
    将图片转换为1X1024的vector，用Numpy的array存储
    :param filename:
    :return:
    '''
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handWrirtingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0] #去掉 ' .txt'
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i : ] = img2Vector('trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('testDigits/%s ' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print 'the classifier came back with: %d, the real answer is: %d' % (classifierResult, classNumStr)
        if classifierResult != classNumStr:
            errorCount += 1.0
        print "\nthe total number of errors is: %d " % (errorCount)
        print "\nthe total error rate is: %f" %(errorCount / float(mTest))



