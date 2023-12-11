import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from util.config import OptionConf
from util.dataSplit import *
from util.io import FileIO
from model.MHCN import MHCN


# QRec函数工作为加载原始输入数据
class QRec(object):
    def __init__(self, config):
        self.trainingData = []
        self.userArtistData = []
        self.userTagData = []
        self.whetherArtistOrTag = 1  # 默认读取的为Artist
        self.testData = []
        self.relation = []
        self.measure = []
        self.config = config
        self.ratingConfig = OptionConf(config['ratings.setup'])
        self.binarized = True
        self.evaluation = OptionConf(config['evaluation.setup'])
        self.results = []
        self.bestParameter = []
        bottom = float(self.evaluation['-b'])
        # 尝试加载UserXXXX数据
        try:
            self.userArtistData = FileIO.loadDataSet(config, config['user_artist_data'], binarized=self.binarized,
                                                     threshold=bottom)
            self.whetherArtistOrTag = 1
        except:
            self.userTagData = FileIO.loadDataSet(config, config['user_tag_data'], binarized=self.binarized,
                                                  threshold=bottom)
            self.whetherArtistOrTag = 0
        self.trainingData = self.userArtistData if self.userArtistData else self.userTagData
        self.socialConfig = OptionConf(self.config['social.setup'])
        # 加载关系文件，储存在relation中
        self.relation = FileIO.loadRelationship(config, self.config['social'])
        print('Reading data and preprocessing...')


    def train_and_evaluate(self):
        # 检测k是否合格
        k = int(self.evaluation['-cv'])
        if k < 2 or k > 10:  # limit to 2-10 fold cross validation
            print("k for cross-validation should not be greater than 10 or less than 2")
            exit(-1)

        # 训练
        i = 1
        for train, test in DataSplit.crossValidation(self.trainingData, k, binarized=self.binarized):
            fold = '[' + str(i) + ']'
            myMHCN = MHCN(self.config, train, test, self.relation, fold, self.whetherArtistOrTag, i)
            myMHCN.execute()
            i += 1

    def predict(self):
        myMHCN = MHCN(self.config, self.trainingData,None, self.relation,None, self.whetherArtistOrTag,None)
        myMHCN.execute()


