import numpy as np
from util.config import OptionConf
import random
from collections import defaultdict
class Rating(object):
    'data access control'
    def __init__(self,config,trainingSet, testSet=None):
        self.config = config
        self.evalSettings = OptionConf(self.config['evaluation.setup'])
        self.user = {} # map user names to id
        self.item = {} # map item names to id
        self.id2user = {}
        self.id2item = {}
        self.userMeans = {} # mean values of users's ratings
        self.itemMeans = {} # mean values of items's ratings
        self.globalMean = 0
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict)  # test set in the form of [user][item]=rating
        self.testSet_i = defaultdict(dict)  # test set in the form of [item][user]=rating
        self.rScale = [] # rating scale
        self.trainingData = trainingSet[:]
        try:
            self.testData = testSet[:]
            self.__generateSet()
        except TypeError:
            self.testData = None
            self.__generateSetForPredict()

        self.__computeItemMean()
        self.__computeUserMean()
        self.__globalAverage()


    def __generateSet(self):
        scale = set()
        for i,entry in enumerate(self.trainingData):
            userName,itemName,rating = entry
            # 给用户再次编号，方便从关系矩阵中取出
            if userName not in self.user:
                self.user[userName] = len(self.user)
                self.id2user[self.user[userName]] = userName
            # 给物品再次编号，方便从关系矩阵中去除
            if itemName not in self.item:
                self.item[itemName] = len(self.item)
                self.id2item[self.item[itemName]] = itemName
            self.trainSet_u[userName][itemName] = rating
            self.trainSet_i[itemName][userName] = rating
            scale.add(float(rating))
        self.rScale = list(scale)
        self.rScale.sort()
        for entry in self.testData:
            userName, itemName, rating = entry
            self.testSet_u[userName][itemName] = rating
            self.testSet_i[itemName][userName] = rating

    def __generateSetForPredict(self):
        scale = set()
        # 遍历训练数据
        for i, entry in enumerate(self.trainingData):
            userName, itemName, rating = entry
            # 如果用户不在字典中，将其加入字典，并记录用户索引
            if userName not in self.user:
                self.user[userName] = len(self.user)
                self.id2user[self.user[userName]] = userName
            # 如果物品不在字典中，将其加入字典，并记录物品索引
            if itemName not in self.item:
                self.item[itemName] = len(self.item)
                self.id2item[self.item[itemName]] = itemName
            # 将评分信息添加到训练集中
            self.trainSet_u[userName][itemName] = rating
            self.trainSet_i[itemName][userName] = rating
            # 将评分值加入到集合中，用于后续计算评分的范围
            scale.add(float(rating))
        # 对评分范围进行排序
        self.rScale = list(scale)
        self.rScale.sort()


    def __globalAverage(self):
        total = sum(self.userMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.userMeans)

    def __computeUserMean(self):
        for u in self.user:
            self.userMeans[u] = sum(self.trainSet_u[u].values())/len(self.trainSet_u[u])

    def __computeItemMean(self):
        for c in self.item:
            self.itemMeans[c] = sum(self.trainSet_i[c].values())/len(self.trainSet_i[c])

    def getUserId(self,u):
        if u in self.user:
            return self.user[u]

    def getItemId(self,i):
        if i in self.item:
            return self.item[i]

    def trainingSize(self):
        return (len(self.user),len(self.item),len(self.trainingData))

    def testSize(self):
        return (len(self.testSet_u),len(self.testSet_i),len(self.testData))

    def contains(self,u,i):
        'whether user u rated item i'
        if u in self.user and i in self.trainSet_u[u]:
            return True
        else:
            return False

    def containsUser(self,u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def containsItem(self,i):
        'whether item is in training set'
        if i in self.item:
            return True
        else:
            return False

    def \
            userRated(self,u):
        return list(self.trainSet_u[u].keys()),list(self.trainSet_u[u].values())

    def itemRated(self,i):
        return list(self.trainSet_i[i].keys()),list(self.trainSet_i[i].values())

    def row(self,u):
        k,v = self.userRated(u)
        vec = np.zeros(len(self.item))
        #print vec
        for pair in zip(k,v):
            iid = self.item[pair[0]]
            vec[iid]=pair[1]
        return vec

    def col(self,i):
        k,v = self.itemRated(i)
        vec = np.zeros(len(self.user))
        #print vec
        for pair in zip(k,v):
            uid = self.user[pair[0]]
            vec[uid]=pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user),len(self.item)))
        for u in self.user:
            k, v = self.userRated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]]=vec
        return m
    # def row(self,u):
    #     return self.trainingMatrix.row(self.getUserId(u))
    #
    # def col(self,c):
    #     return self.trainingMatrix.col(self.getItemId(c))

    def sRow(self,u):
        return self.trainSet_u[u]

    def sCol(self,c):
        return self.trainSet_i[c]

    def rating(self,u,c):
        if self.contains(u,c):
            return self.trainSet_u[u][c]
        return -1

    def ratingScale(self):
        return (self.rScale[0],self.rScale[1])

    def elemCount(self):
        return len(self.trainingData)
