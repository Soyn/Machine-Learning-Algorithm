#-*- coding: utf-8 -*-
__author__ = 'dell'

'绘制二叉树的图'

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''
    创建plot节点（实际上绘制节点）
    '''
    createPlot.ax1.annotate(nodeTxt, xy = parentPt,
    xycoords = "axes fraction", xytext = centerPt, \
    textcoords = "axes fraction", va = "center", ha = "center",bbox = nodeType, arrowprops = arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor= 'pink')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon = False)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def getNumLeafs(myTree):
    '''
    遍历myTree获得叶节点总数
    '''
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    '''
    获取树的深度
    '''
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    '''
    返回一颗树，来测试程序
    '''
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0:'no', 1: 'yes'}}}},
    {'no surfacing': {0: 'no', 1: {'flippers': {0:  {'head': {0: 'no', 1: 'yes'}},1: 'no' }}}}
    ]
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt, txtString):
    '''
    在父子节点之间绘制文本信息
    '''
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    '''
    获得宽度和高度
    '''
    numLeafs = getNumLeafs(myTree)
    getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xoff + float(numLeafs) / 2.0 / plotTree.totalW, plotTree.yoff)
    #绘制字节点的值：
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    #减少Y的偏移：
    secondDict = myTree[firstStr]
    plotTree.yoff = plotTree.yoff - 1.0 / plotTree.tatalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
            plotTree(secondDict[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)
            plotMidText((plotTree.xoff ,plotTree.yoff), cntrPt, str(key))
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD

def createTree(inTree):
    fig = plt.figure(1, facecolor='pink')
    cig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xoff = -0.5 / plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

if __name__ == '__main__':
    createPlot()
    '''print retrieveTree(1)
    myTree = retrieveTree(0)
    print getNumLeafs(myTree)
    print getTreeDepth(myTree)'''






