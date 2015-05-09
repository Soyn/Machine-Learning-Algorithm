#-*- coding: utf-8 -*-
__author__ = 'dell'
'朴素贝叶斯分类器'


import numpy as np
import math
import random as rd

import feedparser
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
    '''
    测试分类器
    :return:
    '''
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

def bagOfWords2VecMN(vocabList, inputSet):
    '''
    将多个出现的
    :param vocabList:
    :param inputSet:
    :return:
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            #将出现的词加1
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString):
    '''
    利用正则表达式匹配字符串，生成词条，用list存储
    :param bigString:
    :return:
    '''
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2 ]



def  spamTest():
    '''
    对贝叶斯邮件分类器进行自动化处理
    实现邮件过滤功能
    :return:
    '''
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        #导入文件并将其解析为词列表
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        #随机生成randIndex
        randIndex = int(rd.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        print testSet
        del(trainingSet[randIndex])


    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #计算trainingSet的概率
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

    errorCount = 0
    for docIndex in testSet:
        #计算错误率
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])#构建词向量
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]
    print 'the error rate is: ', float(errorCount) / len(testSet)

#利用朴素贝叶斯分类器从个人广告中筛选出地域倾向
def calcMostFreq(vocabList, fullText):
    '''
    计算出现的频率
    :param vocabList:
    :param fullText:
    :return:
    返回30个出现频率最高的单词的
    '''
    import operator
    freqDict = {}
    for token in vocabList:
        #遍历vocablist，统计在text中出现的词汇
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser
    doclist = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))

    for i in range(minLen):
        #每次访问一条rss源
        wordList = textParse(feed1['entries'][i]['summary'])
        doclist.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        doclist.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(doclist)
    top30Words = calcMostFreq(vocabList, fullText)

    for pairW in top30Words:
        if pairW[0] in vocabList:
            #删除单词表中出现频率最高的词剔除
            vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen)
    testSet = []
    for i in range(20):
        randIndex = int(rd.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, doclist[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, doclist[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float (errorCount) / len(testSet)
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    '''
    显示地域相关的用词
    按照顺序输出pSF和 pNY的内容
    :param ny,sf:
     本地解析的feed源
    :return:
    '''
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)#训练并测试分类器
    topNy = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0 :
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > - 6.0:
            topNy.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key = lambda pair: pair[1], reverse = True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNy, key = lambda pair: pair[1], reverse = True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY **"
    for item in sortedNY:
        print item[0]



