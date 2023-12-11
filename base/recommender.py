import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import json
import os
import numpy as np
from data.rating import Rating
from util.io import FileIO
from util.config import OptionConf
from util.log import Log
from os.path import abspath
from time import strftime,localtime,time
from util.measure import Measure
from util.qmath import find_k_largest


class Recommender(object):
    def __init__(self,conf,trainingSet,testSet,fold,whetherArtistOrTag=1):
        self.config = conf
        self.data = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.isOutput = True
        self.data = Rating(self.config, trainingSet, testSet)
        self.foldInfo = fold
        self.evalSettings = OptionConf(self.config['evaluation.setup'])
        self.measure = []
        self.recOutput = []
        self.num_users, self.num_items, self.train_size = self.data.trainingSize()
        self.whetherArtistOrTag = whetherArtistOrTag
    def initializing_log(self):
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        try:
            self.log = Log(self.modelName, self.modelName + self.foldInfo + ' ' + currentTime)
            # save configuration
            self.log.add('### model configuration ###')
            for k in self.config.config:
                self.log.add(k + '=' + self.config[k])
        except:
            self.log = None

    def readConfiguration(self):
        self.modelName = self.config['model.name']
        self.output = OptionConf(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()
        self.ranking = OptionConf(self.config['item.ranking'])

    def printAlgorConfig(self):
        "show model's configuration"
        print('Model:',self.config['model.name'])
        if OptionConf(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:', abspath(OptionConf(self.config['evaluation.setup'])['-testSet']))
        #print dataset statistics
        print('Training set size: (user count: %d, item count %d, record count: %d)' %(self.data.trainingSize()))
        try:
            print('Test set size: (user count: %d, item count %d, record count: %d)' %(self.data.testSize()))
        except:
            print('No Test Data')
        print('='*80)
        #print specific parameters if applicable
        if self.config.contains(self.config['model.name']):
            parStr = ''
            args = OptionConf(self.config[self.config['model.name']])
            for key in args.keys():
                parStr+=key[1:]+':'+args[key]+'  '
            print('Specific parameters:',parStr)
            print('=' * 80)

    def initModel(self):
        pass

    def trainModel(self):
        'build the model (for model-based Models )'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    #for rating prediction
    def predictForRating(self, u, i):
        pass

    #for item prediction
    def predictForRanking(self,u):
        pass

    def checkRatingBoundary(self,prediction):
        if prediction > self.data.rScale[-1]:
            return self.data.rScale[-1]
        elif prediction < self.data.rScale[0]:
            return self.data.rScale[0]
        else:
            return round(prediction,3)

    def evalRanking(self):
        # top几的检验
        if self.ranking.contains('-topN'):
            top = self.ranking['-topN'].split(',')
            top = [int(num) for num in top]
            N = max(top)
            if N > 100 or N < 1:
                print('N can not be larger than 100! It has been reassigned to 10')
                N = 10
        else:
            print('No correct evaluation metric is specified!')
            exit(-1)

        # 获取推荐名单(test版)
        recList = self.SaveRecList(N)

        # 写模型参数对应性能
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        outDir = self.output['-dir']
        fileName = self.config['model.name'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        self.measure = Measure.rankingMeasure(self.data.testSet_u, recList, top)
        FileIO.writeFile(outDir, fileName, self.measure)
        print('The result of %s %s:\n%s' % (self.modelName, self.foldInfo, ''.join(self.measure)))
        #写日志
        self.log.add('###Evaluation Results###')
        self.log.add(self.measure)

    def SaveRecList(self,N):
        # 获取推荐名单 recList
        recList = {}
        userCount = len(self.data.testSet_u)
        for i, user in enumerate(self.data.testSet_u):
            line = user + ':'
            candidates = self.predictForRanking(user)
            ratedList, ratingList = self.data.userRated(user)
            for item in ratedList:
                candidates[self.data.item[item]] = 0
            ids, scores = find_k_largest(N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            recList[user] = list(zip(item_names, scores))

        # 保存 recList
        script_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(script_directory)
        folder_path = os.path.join(parent_directory, 'SaveSelf')
        os.makedirs(folder_path, exist_ok=True)  # 创建文件夹，如果不存在的话
        # 看看是用的哪一个物品
        if(self.whetherArtistOrTag):
            file_path = os.path.join(folder_path, 'recListArtist.json')
            np.savetxt("KmeansData/ArtistEmbedding.txt", self.V, fmt='%f', delimiter='\t')
            # 打开文件进行写操作
            with open('./KmeansData/IdArtist.txt', 'w') as file:
                # 遍历self.data.id2item字典的值并将实际的列编号写入文件
                for item_id, item_code in self.data.id2item.items():
                    file.write(str(item_code) + '\n')
        else:
            file_path = os.path.join(folder_path, 'recListTag.json')
            np.savetxt("KmeansData/TagEmbedding.txt", self.V, fmt='%f', delimiter='\t')
            with open('./KmeansData/IdTag.txt', 'w') as file:
                # 遍历self.data.id2item字典的值并将实际的列编号写入文件
                for item_id, item_code in self.data.id2item.items():
                    file.write(str(item_code) + '\n')

        return recList

    def execute(self):
        self.initializing_log()
        self.printAlgorConfig()

        # build model and train
        if(self.foldInfo):
            print('Building Model %s...' % self.foldInfo)
            # 训练模型
            self.trainModel()
            # rating prediction or item ranking
            print('Predicting %s...' % self.foldInfo)
            # 评估性能，获得推荐名单
            self.evalRanking()

        # predict
        else:
            print('Loading model ...')
            self.predict()
            self.get_top_items_for_users()

    def compute_user_preferences(self, num_top_items=10):
        user_preferences = np.dot(self.P, self.Q.T)

        # 仅保留每行最大的几个值
        top_items_indices = np.argpartition(user_preferences, -num_top_items, axis=1)[:, -num_top_items:]
        user_preferences = np.take_along_axis(user_preferences, top_items_indices, axis=1)

        return user_preferences, top_items_indices

    def get_top_items_for_users(self, num_top_items=10):
        # 计算用户对物品的喜好程度和相应的索引
        user_preferences, top_items_indices = self.compute_user_preferences(num_top_items)

        # 转换物品的索引为实际的ID
        top_items = np.vectorize(lambda x: self.data.id2item[x])(top_items_indices)  # 假设物品的标识是从1开始的

        # 保存结果到文件夹
        save_path="Save"
        if(self.whetherArtistOrTag==1):
            self.save_results(top_items, 'recArtistList.json')
        else:
            self.save_results(top_items, 'recTagList.json')
        self.save_results(self.P, 'P.npy')
        self.save_results(self.Q, 'Q.npy')
        self.save_results(self.data.id2user, 'id2user.json')
        self.save_results(self.data.id2item, 'id2item.json')

        return top_items

    def save_results(self, data, filename):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(script_directory)
        folder_path = os.path.join(parent_directory, 'SaveResults')
        os.makedirs(folder_path, exist_ok=True)  # 创建文件夹，如果不存在的话
        file_path = os.path.join(folder_path, filename)
        if isinstance(data, np.ndarray):
            np.save(file_path, data)
        elif isinstance(data, dict):
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file)
        else:
            raise ValueError("Unsupported data type for saving.")













