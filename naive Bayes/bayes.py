#-*- coding: utf-8 -*-
__author__ = 'dell'
'朴素贝叶斯分类器'


import numpy as np
import math

def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1] #1 表示侮辱性的词汇，0表示正常
    return postingList, classVec

def createVocabList(dataSet):
    #创建无重复词的词汇表
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创建两个集合的并集
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    #遍历文件中的单词，如果输入的词中含有单词，将含有单词的索引出设置为1
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my VocabList!" % word
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    '''
    朴素贝叶斯分类器训练函数
    :param trainMatrix:
    文档的内容为矩阵
    :param trainCategory:
    每篇文档构成的的向量trainCategory
    :return:
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    #生成元素值为0的矩阵
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)

    p0Denom = 2.0
    p1Denom = 2.0

    #遍历所有文件，对出现在在文件中的词条计数，最后总的词条也要加1:
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    #计算概率：
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''

    :param vec2Classify:
    需要分类的向量
    :param p0Vec:,p1Vecp,Class1:
    为trainNB0计算的概率
    :return:
    '''
    #vec2Classify中的元素与p1Vec中的对应相乘后加上词汇表中的值相加
    p1 = sum(vec2Classify * p1Vec, ) +np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat),np. array(listClasses))
    testEntry = ['love', 'my', 'dalmation', 'stupid', 'dog', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage', 'love', 'my']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)

if __name__ == '__main__':
     testingNB()