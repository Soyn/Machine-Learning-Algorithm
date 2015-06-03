#-*- coding:utf8 -*-
__author__ = 'dell'

'做出回归树的图形界面'

from numpy import *

from Tkinter import *

import regTree
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS, tolN):
    '''
    改变参数后重新绘图
    :param tolS:
    :param tolN:
    :return:
    '''
    reDraw.f.clf() #清除掉上一次的图像
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = regTree.createTree(reDraw.rawDat, regTree.modelLeaf, regTree.modelErr, (tolS, tolN))
        yHat = regTree.createForeCast(myTree, reDraw.testDat, regTree.modelTreeEval)
    else:
        myTree = regTree.createTree(reDraw.rawDat, ops = (tolS, tolN))
        yHat = regTree.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0], reDraw.rawDat[:, 1], s = 5)
    reDraw.a.plot(reDraw.testDat, yHat, linewidth = 2.0)
    reDraw.canvas.show()

def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print "enter Integer for tolN"
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print "enter Float for tols."
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS



def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)

root = Tk()

reDraw.f = Figure(figsize = (5, 4), dpi = 100) #创建画布
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master = root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row = 0, columnspan = 3)

Label(root, text = "Plot Place Holder").grid(row = 0, columnspan = 3)

Label(root, text = "tolN").grid(row = 1, column = 0)
tolNentry = Entry(root)
tolNentry.grid(row = 1, column = 1)
tolNentry.insert(0, '10')

Label(root, text = "tolS").grid(row = 2, column = 0)
tolSentry = Entry(root)
tolSentry.grid(row = 2, column = 1)
tolSentry.insert(0, '1.0')

Button(root, text = "ReDraw", command = drawNewTree).grid(row = 1, column = 2, rowspan = 3)
chkBtnVar = IntVar()#标识CheckButton的状态
Button(root, text = "Quit", fg = "black", command = root.quit).grid(row = 1, column = 2)
chkBtn = Checkbutton(root, text = "Model Tree", variable = chkBtnVar)
chkBtn.grid(row = 3, column = 0, columnspan = 2)

reDraw.rawDat = mat(regTree.loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)

reDraw(1.0, 10)

root.mainloop()









