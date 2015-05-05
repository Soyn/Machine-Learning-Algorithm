#-*- coding: utf-8 -*-
__author__ = 'dell'
'将文件转换成词汇表中的词向量'

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

if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print myVocabList
    print setOfWords2Vec(myVocabList, listOPosts[0])