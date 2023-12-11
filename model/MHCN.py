import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import os
from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
from scipy.sparse import coo_matrix
from util.loss import bpr_loss
from util import config
from math import sqrt
class MHCN(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold=None, whetherArtistOrTag=True, k=None):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold,whetherArtistOrTag=whetherArtistOrTag)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold,whetherArtistOrTag=whetherArtistOrTag)
        self.whetherArtistOrTag = whetherArtistOrTag
        self.k = k
        self.readConfiguration()
        if(self.foldInfo==None):
            self.initModel()
            self.loadModel()
        else:
            print('Initializing model %s...' % self.foldInfo)
            self.initModel()

    def readConfiguration(self):
        super(MHCN, self).readConfiguration()
        args = config.OptionConf(self.config['MHCN'])
        self.n_layers = int(args['-n_layer'])
        self.ss_rate = float(args['-ss_rate'])
    def initModel(self):
        super(MHCN, self).initModel()
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        self.n_channel = 4
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        # define learnable paramters
        for i in range(self.n_channel):
            self.weights['gating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]),
                                                             name='g_W_%d_1' % (i + 1))
            self.weights['gating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]),
                                                                  name='g_W_b_%d_1' % (i + 1))
            self.weights['sgating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]),
                                                              name='sg_W_%d_1' % (i + 1))
            self.weights['sgating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]),
                                                                   name='sg_W_b_%d_1' % (i + 1))
        self.weights['attention'] = tf.Variable(initializer([1, self.emb_size]), name='at')
        self.weights['attention_mat'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='atm')

        self.M_matrices = self.buildMotifInducedAdjacencyMatrix()
        # initialize adjacency matrices
        self.H_s = self.M_matrices[0]
        self.H_s = self.adj_to_sparse_tensor(self.H_s)
        self.H_j = self.M_matrices[1]
        self.H_j = self.adj_to_sparse_tensor(self.H_j)
        self.H_p = self.M_matrices[2]
        self.H_p = self.adj_to_sparse_tensor(self.H_p)
        self.R = self.buildJointAdjacency()

        self.ss_loss = 0

    def loadModel(self):
        # 获取新模型的可训练参数列表
        new_train_vars = tf.trainable_variables()
        # 从第三个变量开始获取需要加载的变量
        vars_to_load = new_train_vars[2:]
        # 创建 Saver 对象，指定要加载的变量列表
        saver = tf.train.Saver(var_list=vars_to_load)
        # 构建加载路径
        load_path = "model/Parameter/第{}折/第{}epoch/model".format(1, 1)
        # 创建 TensorFlow 会话
        with tf.Session() as sess:
            # 使用 Saver 对象加载模型参数
            saver.restore(sess, load_path)
    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix
    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return ratingMatrix
    def buildJointAdjacency(self):
        indices = [[self.data.user[item[0]], self.data.item[item[1]]] for item in self.data.trainingData]
        values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
                  for item in self.data.trainingData]
        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.num_users, self.num_items])
        return norm_adj
    def buildMotifInducedAdjacencyMatrix(self):
        S = self.buildSparseRelationMatrix()
        Y = self.buildSparseRatingMatrix()
        self.userAdjacency = Y.tocsr()
        self.itemAdjacency = Y.T.tocsr()
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        A8 = (Y.dot(Y.T)).multiply(B)
        A9 = (Y.dot(Y.T)).multiply(U)
        A9 = A9+A9.T
        A10  = Y.dot(Y.T)-A8-A9
        #addition and row-normalization
        H_s = sum([A1,A2,A3,A4,A5,A6,A7])
        H_s = H_s.multiply(1.0/H_s.sum(axis=1).reshape(-1, 1))
        H_j = sum([A8,A9])
        H_j = H_j.multiply(1.0/H_j.sum(axis=1).reshape(-1, 1))
        H_p = A10
        H_p = H_p.multiply(H_p>1)
        H_p = H_p.multiply(1.0/H_p.sum(axis=1).reshape(-1, 1))

        return [H_s,H_j,H_p]
    def adj_to_sparse_tensor(self,adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj
    def hierarchical_self_supervision(self,em,adj):
        def row_shuffle(embedding):
            return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))
        def row_column_shuffle(embedding):
            corrupted_embedding = tf.transpose(tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0]))))
            corrupted_embedding = tf.gather(corrupted_embedding, tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
            return corrupted_embedding
        def score(x1,x2):
            return tf.reduce_sum(tf.multiply(x1,x2),1)
        user_embeddings = em
        # user_embeddings = tf.math.l2_normalize(em,1) #For Douban, normalization is needed.
        edge_embeddings = tf.sparse_tensor_dense_matmul(adj,user_embeddings)
        #Local MIM
        pos = score(user_embeddings,edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings),edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings),user_embeddings)
        local_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos-neg1))-tf.log(tf.sigmoid(neg1-neg2)))
        #Global MIM
        graph = tf.reduce_mean(edge_embeddings,0)
        pos = score(edge_embeddings,graph)
        neg1 = score(row_column_shuffle(edge_embeddings),graph)
        global_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos-neg1)))
        return global_loss+local_loss

    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items
    def gating(self, em, channel):
        return tf.multiply(em, tf.nn.sigmoid(
            tf.matmul(em, self.weights['gating%d' % channel]) + self.weights['gating_bias%d' % channel]))
    def supervised_gating(self, em, channel):
        return tf.multiply(em, tf.nn.sigmoid(
            tf.matmul(em, self.weights['sgating%d' % channel]) + self.weights['sgating_bias%d' % channel]))
    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(tf.reduce_sum(
                tf.multiply(self.weights['attention'], tf.matmul(embedding, self.weights['attention_mat'])), 1))
        score = tf.nn.softmax(tf.transpose(weights))
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += tf.transpose(tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i])))
        return mixed_embeddings, score

    def model(self):
        # self-gating
        user_embeddings_c1 = self.gating(self.user_embeddings, 1)
        user_embeddings_c2 = self.gating(self.user_embeddings, 2)
        user_embeddings_c3 = self.gating(self.user_embeddings, 3)
        simple_user_embeddings = self.gating(self.user_embeddings, 4)
        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        item_embeddings = self.item_embeddings
        all_embeddings_i = [item_embeddings]
        self.ss_loss = 0
        # multi-channel convolution
        for k in range(self.n_layers):
            mixed_embedding = self.channel_attention(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)[0] + simple_user_embeddings / 2
            # Channel S
            user_embeddings_c1 = tf.sparse_tensor_dense_matmul(self.H_s, user_embeddings_c1)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c1, axis=1)
            all_embeddings_c1 += [norm_embeddings]
            # Channel J
            user_embeddings_c2 = tf.sparse_tensor_dense_matmul(self.H_j, user_embeddings_c2)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c2, axis=1)
            all_embeddings_c2 += [norm_embeddings]
            # Channel P
            user_embeddings_c3 = tf.sparse_tensor_dense_matmul(self.H_p, user_embeddings_c3)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c3, axis=1)
            all_embeddings_c3 += [norm_embeddings]
            # item convolution
            new_item_embeddings = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(self.R), mixed_embedding)
            norm_embeddings = tf.math.l2_normalize(new_item_embeddings, axis=1)
            all_embeddings_i += [norm_embeddings]
            simple_user_embeddings = tf.sparse_tensor_dense_matmul(self.R, item_embeddings)
            all_embeddings_simple += [tf.math.l2_normalize(simple_user_embeddings, axis=1)]
            item_embeddings = new_item_embeddings
        # averaging the channel-specific embeddings
        user_embeddings_c1 = tf.reduce_sum(all_embeddings_c1, axis=0)
        user_embeddings_c2 = tf.reduce_sum(all_embeddings_c2, axis=0)
        user_embeddings_c3 = tf.reduce_sum(all_embeddings_c3, axis=0)
        simple_user_embeddings = tf.reduce_sum(all_embeddings_simple, axis=0)
        item_embeddings = tf.reduce_sum(all_embeddings_i, axis=0)
        # aggregating channel-specific embeddings
        self.final_item_embeddings = item_embeddings
        self.final_user_embeddings, self.attention_score = self.channel_attention(user_embeddings_c1, user_embeddings_c2,user_embeddings_c3)
        self.final_user_embeddings += simple_user_embeddings / 2
        # create self-supervised loss
        self.ss_loss += self.hierarchical_self_supervision(self.supervised_gating(self.final_user_embeddings, 1), self.H_s)
        self.ss_loss += self.hierarchical_self_supervision(self.supervised_gating(self.final_user_embeddings, 2), self.H_j)
        self.ss_loss += self.hierarchical_self_supervision(self.supervised_gating(self.final_user_embeddings, 3), self.H_p)
        # embedding look-up
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)

    def trainModel(self):
        self.model()
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        reg_loss = 0
        for key in self.weights:
            reg_loss += 0.001 * tf.nn.l2_loss(self.weights[key])
        reg_loss += self.regU * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        total_loss = rec_loss + reg_loss + self.ss_rate * self.ss_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train_vars = tf.trainable_variables()  # 获取可训练变量列表
        train_op = opt.minimize(total_loss, var_list=train_vars)  # 指定优化器只优化可训练变量
        init = tf.global_variables_initializer()
        self.sess.run(init)
        bestLoss=float('inf')
        best_train_vars = None
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l1 = self.sess.run([train_op, rec_loss],feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print(self.foldInfo, 'training:', epoch + 1, 'batch', n, 'rec loss:', l1)
            if(bestLoss>l1):
                    # 获取所有可训练变量
                    best_train_vars=train_vars
                    # 从第三个变量到最后一个变量
                    variables_to_save = best_train_vars[2:]
                    # 构建保存路径
                    save_path = "model/Parameter/第{}折/epoch:{}/".format(self.k, epoch + 1)
                    # 使用 os.makedirs 创建路径（确保路径存在）
                    os.makedirs(save_path, exist_ok=True)
                    # 创建 Saver 对象，指定要保存的变量列表
                    saver = tf.train.Saver(var_list=variables_to_save)
                    # 使用 Saver 对象保存模型参数
                    saver.save(self.sess, os.path.join(save_path, "model"))
            self.U, self.V = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])
            self.ranking_performance(epoch)
        self.U, self.V = self.bestU, self.bestV

    def predict(self):
        self.model()




