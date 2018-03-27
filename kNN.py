#coding=UTF8
from numpy import *
import operator
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import re

global datalength
f1=open(r'datingTestSet2.txt')
arrayOlines = f1.readlines()  #每行的样本数据
datalength=len(re.split(" |!|\?|\.|,|\t",arrayOlines[0]))-1
f1.close()


def classify0(inX, dataset, labels, k):
    """
    inX 是输入的测试样本，是一个[x, y]样式的
    dataset 是训练样本集
    labels 是训练样本标签
    k 是top k最相近的
    """
    # shape返回矩阵的[行数，列数]，
    # 行数就是样本的数量
    dataSetSize = dataset.shape[0] #shape[0]获取数据集的行数   
    diffMat = tile(inX, (dataSetSize, 1)) - dataset #把输入的测试样本扩展为和dataset的sharp一样，然后直接做矩阵减法

    sqDiffMat = diffMat ** 2 #矩阵平方，即对矩阵中的每个元素进行平方操作
    sqDistance = sqDiffMat.sum(axis=1) #axis表示横轴，按照行进行累加    
    distance = sqDistance ** 0.5 # 对平方和进行开根号
    sortedDistIndicies = distance.argsort()  # 按照升序进行快速排序，返回的是原数组的下标
    classCount = {} # 存放最终的分类结果及相应的结果投票数
    
    # 投票过程，就是统计前k个最近的样本所属类别包含的样本个数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #对应的分类结果
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # classCount.get(voteIlabel, 0)返回voteIlabel的值，如果不存在，则返回0，然后将票数增1
    
    # 把分类结果进行排序，然后返回得票数最多的分类结果
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
def file2matrix(filename):
    """
    从文件中读入训练数据，并存储为矩阵
    """
    fr = open(filename)
    arrayOlines = fr.readlines()  #每行的样本数据
    global numberOfLines
    numberOfLines = len(arrayOlines)   #获取样本的行数
    returnMat = zeros((numberOfLines,datalength))   #创建一个0矩阵用于存放训练样本数据，一共有n行，每一行存放相应个数的数据
    fr.close()
    classLabelVector = []    #创建一个1维数组用于存放训练样本标签。  
    index = 0
    for line in arrayOlines:
        # 把回车符号给去掉
        line = line.strip()
        # 把每一行数据用\t分割
        listFromLine = re.split(" |!|\?|\.|,|\t",line)
        # 把分割好的数据放至数据集，其中index是该样本数据的下标，就是放到第几行
        returnMat[index,:] = listFromLine[0:datalength]#将每一个样本的所有参数放在一个列表内
        # 把该样本对应的分类标签放至标签集，顺序与样本集对应。
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
    
def autoNorm(dataSet):
    """
    训练数据归一化
    """  
    minVals = dataSet.min(0) # 获取数据集中每一列的最小数值
    maxVals = dataSet.max(0) # 获取数据集中每一列的最大数值

    ranges = maxVals - minVals # 最大值与最小的差值
    # 创建一个与dataSet同shape的全0矩阵，用于存放归一化后的数据
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 把最小值扩充为与dataSet同shape，然后作差
    normDataSet = dataSet - tile(minVals, (m,1))#tile将最小值在行上重复m次，在列上重复1次
    # 把最大最小差值扩充为dataSet同shape，然后作商，是指对应元素进行除法运算，而不是矩阵除法。
    
    normDataSet = normDataSet/tile(ranges, (m,1))
    
    return normDataSet, ranges, minVals
   
def datingClassTest():
    # 将数据集中10%的数据留作测试用，其余的90%用于训练  
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]#m是矩阵的行数
    numTestVecs = int(m*hoRatio)
    errordict={}
    
    for k in range(1,int(numberOfLines**0.5),2):#找出最合适的k值
        errorCount=0.0
        for i in range(numTestVecs):
            classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],k)#输入的测试样本（是一个[x, y]样式的），训练样本集，训练样本标签，k值
        
            if (classifierResult != datingLabels[i]):
                errorCount += 1.0
        errordict[k]=errorCount

    k_min=min(errordict, key=errordict.get)#错误率最小的k值
    print("The value of key is:",k_min)
    error_Count=0.0
    for i in range(numTestVecs):
            classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],k_min)#输入的测试样本（是一个[x, y]样式的），训练样本集，训练样本标签，k值
            print("the classifier came back with: %d, the real answer is: %d, result is :%s" % (classifierResult, datingLabels[i],classifierResult==datingLabels[i]))
            if (classifierResult != datingLabels[i]):
                error_Count += 1.0
   
    print("the total error rate is: %f" % (error_Count/float(numTestVecs)))
    print("errorCount is:",error_Count)
    



if __name__== "__main__":  

    datingClassTest()


 
    
