#-*- coding: utf-8 -*-
__author__ = 'dell'

'SVM算法的简化版本SMO'

import random as nd
import numpy as np
def loadDataSet(filename):
    '''
    从文件中读取数据
    :param filename:
    :return:
    以列表形式返回数据矩阵，和类别
    '''
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    '''

    :param
    i: alpha的索引
    :param m:
    alpha的数量
    :return:
    '''
    j = 1
    while(j == i):
        j = int(nd.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    SMO算法
    :param dataMatIn:
     数据集
    :param classLabels:
     类别标签
    :param C:
     C为常量
    :param toler:
     toler为容错
    :param maxIter:
     最大迭代次数
    :return:
    '''
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()#转置矩阵
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0#没有任何alphas改变的情况下遍历数据集的次数

    while(iter < maxIter):
        alphaPairsChanged = 0#记录alphas是否优化了
        for i in range(m):

            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b#预测类
            Ei = fXi -float(labelMat[i])#计算误差

            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C) ) or \
                    ((labelMat[i] * Ei > toler) and (alphas[i] > 0 )): #当alpha可以优化时

                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat). T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])

                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #L和H来用于将label[i]限制在0~C
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C,  alphas[j] + alphas[i])

                if L == H:print "L == H" ;continue

                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T

                if eta >= 0:   print "eta>=0";continue
                alphas[j] -= labelMat[j] * (Ei - Ej) /eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                if(np.abs(alphas[j] - alphaJold) < 0.00001):

                    print "j not moving enough";continue

                alphas[i]  += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T -labelMat[j] * (alphas[j] - alphaJold)*dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) *dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[j]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print "iter: %d i: %d, pairs changed %d " % (iter, i, alphaPairsChanged)

        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" %iter
    return b,alphas


def calcEk(oS, K):
        '''
        每给一个alpha计算一个E值
        :param oS:
        :param K:
        :return:
        '''
        fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,K] + oS.b)
        Ek = fXk - float(oS.labelMat[K])
        return Ek

def selectJ(i, oS, Ei):


    '''
    获取第二个alpha值或者说是内循环的alpha值
    :param i:
    :param oS:
    :param Ei:
    第一个alpha的错误
    :return:
    '''
    maxK = -1
    maxDeltae = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    #nonzeros（）返回非0E值对应的alphas值
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]

    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = np.abs(Ei - Ek)
            if deltaE > maxDeltae:#选择j的最大步长
                maxK = k
                maxDeltae = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):

    '''
    计算误差将其放入到cache中
    :param oS:
    :param k:
    :return:
    '''
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):

    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    '''
    优化后的SMO算法
    :param dataMatIn:
    :param classLabels:
    :param C:
    :param toler:
    :param maxIter:
    :param kTup:
    :return:
    '''
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler,kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    while (iter < maxIter) and ((alphaPairsChanged) > 0 or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS) #选择第二个alpha
            print "fullSet, iter: %d i: %d, pairs changed %d"  % (iter, i, alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A) > 0 * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print "non-bound, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print "Iteration number: %d" % iter
    return oS.b, oS.alphas

def clacWs(alphas, dataArr, classLabels):
    '''
    获得超平面
    :param alphas:
    :param dataArr:
    :param classLabels:
    :return:
    '''
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

def kernelTrans(X, A, kTup):
    '''
    应用实核函数转换，这里利用的是高斯版本的径向基核函数
    :param X:
    描述实核函数的类型的一个字符串
    :param A:
    :param kTup:
    :return:
    '''
    m,n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            #计算高斯表达式
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T

        K = np.exp(K / (-1 * kTup[1] ** 2))#元素之间相除
    else:
        raise NameError('Houston We have a Problem --__--That Kernel is not recongnized')
    return K

class optStruct(object):
    '''
    定义类来存储数据集
    '''
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        #kTup包含了与实核函数有关的信息
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def testRbf(k1 = 1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print "'there are %d Support Vectors." % np.shape(sVs)[0]
    m,n = np.shape(datMat)
    errorCount = 0
    for  i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
        print "'the training error rate is: %f" % (float (errorCount) / m)
        dataArr, labelArr = loadDataSet('testSetRBF.txt')
        errorCount = 0
        datMat = np.mat(dataArr)
        labelMat = np.mat(labelArr).transpose()
        m, n = np.shape(datMat)
        for i in range(m):
            kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
            predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
            if np.sign(predict) != np.sign(labelArr[i]):
                errorCount += 1
        print "the test error rate is: %f" % (float (errorCount) / m)

def img2Vector(filename):
    '''
    将图片转换为1X1024的vector，用Numpy的array存储
    :param filename:
    :return:
    '''
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import  listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int (fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2Vector('%s/%s ' % (dirName, fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup = ('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print "there are %d Support Vectors." % np.shape(sVs)[0]
    m,n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
        print "the training error rate is: %f " % (float(errorCount) / m)
        dataArr,  labelArr = loadImages('testDigits')
        errorCount = 0
        dataMat = np.mat(dataArr)
        labelMat = np.mat(labelArr).transpose()
        m, n = np.shape(dataMat)
        for i in range(m):
            kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
            predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
            if np.sign(predict) != np.sign(labelArr[i]):
                errorCount += 1
        print "the test error rate is: %f" % (float(errorCount) / m)


if __name__ == "__main__":
    testDigits(('rbf', 20))
