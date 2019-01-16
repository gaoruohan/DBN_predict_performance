"""
Deep Belief Network
author: Cuson
2018/12/16
"""
import timeit
import os

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score,cross_val_predict, GridSearchCV
from BP import BPNeuralNetwork
import tensorflow as tf
import numpy as np
from mlp import HiddenLayer
from rbm import RBM
import random
import matplotlib.pyplot as plt

mm = MinMaxScaler()
def MaxMinNormalization(x):
    mm_data = mm.fit_transform(x)
    return mm_data


# 搭建模型
class DBN(object):
    """
    An implement of deep belief network
    The hidden layers are firstly pretrained by RBM, then DBN is treated as a normal
    MLP by adding a output layer.
    """
    def __init__(self, n_in=784, n_out=10, hidden_layers_sizes=[500, 500]):
        """
        :param n_in: int, the dimension of input
        :param n_out: int, the dimension of output
        :param hidden_layers_sizes: list or tuple, the hidden layer sizes
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # Number of layers
        assert len(hidden_layers_sizes) > 0
        self.n_layers = len(hidden_layers_sizes)
        self.layers = []  # normal sigmoid layer
        self.rbm_layers = []  # RBM layer
        self.params = []  # keep track of params for training

        # Define the input and output
        self.x = tf.placeholder(tf.float32, shape=[None, n_in])
        self.y = tf.placeholder(tf.float32, shape=[None, n_out])

        # Contruct the layers of DBN
        with tf.name_scope('DBN_layer'):
            for i in range(self.n_layers):
                if i == 0:
                    layer_input = self.x
                    input_size = n_in
                else:
                    layer_input = self.layers[i - 1].output
                    input_size = hidden_layers_sizes[i - 1]
                # Sigmoid layer
                with tf.name_scope('internel_layer'):
                    sigmoid_layer = HiddenLayer(input=layer_input, n_in=input_size, n_out=hidden_layers_sizes[i],
                                                activation=tf.nn.sigmoid)
                self.layers.append(sigmoid_layer)
                # Add the parameters for finetuning
                self.params.extend(sigmoid_layer.params)
                # Create the RBM layer
                with tf.name_scope('rbm_layer'):
                    self.rbm_layers.append(RBM(inpt=layer_input, n_visiable=input_size, n_hidden=hidden_layers_sizes[i],
                                               W=sigmoid_layer.W, hbias=sigmoid_layer.b))
            # We use the BP layer as the output layer
            with tf.name_scope('output_layer'):
                self.output_layer = BPNeuralNetwork(inpt=self.layers[-1].output, n_in=hidden_layers_sizes[-1],n_out=n_out)
        self.params.extend(self.output_layer.params)
        # The finetuning cost
        with tf.name_scope('output_loss'):
            self.cost = self.output_layer.cost(self.y)
        # The accuracy
        self.accuracy = self.output_layer.accuarcy(self.y)

    def pretrain(self, sess, train_x, batch_size=2, pretraining_epochs=20, lr=0.01, k=1,
                 display_step=1):
        """
        Pretrain the layers (just train the RBM layers)
        :param sess: tf.Session
        :param X_train: the input of the train set (You might modidy this function if you do not use the desgined mnist)
        :param batch_size: int
        :param lr: float
        :param k: int, use CD-k
        :param pretraining_epoch: int
        :param display_step: int
        """
        print('Starting pretraining...\n')
        start_time = timeit.default_timer()
        # Pretrain layer by layer
        for i in range(self.n_layers):
            cost = self.rbm_layers[i].get_reconstruction_cost()
            train_ops = self.rbm_layers[i].get_train_ops(learning_rate=lr, k=k, persistent=None)
            batch_num = int(train_x.shape[0] / batch_size)

            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for step in range(batch_num - 1):
                    # 训练
                    x_batch = train_x[step * batch_size:(step + 1) * batch_size]

                    sess.run(train_ops, feed_dict={self.x: x_batch})
                    # 计算cost
                    avg_cost += sess.run(cost, feed_dict={self.x: x_batch,})/ batch_num
                    # print(avg_cost)
                # 输出

                if epoch % display_step == 0:
                    print("\tPretraing layer {0} Epoch {1} cost: {2}".format(i, epoch, avg_cost))

        end_time = timeit.default_timer()
        print("\nThe pretraining process ran for {0} minutes".format((end_time - start_time) / 60))

    def finetuning(self, sess, train_x, train_y, test_x, test_y, training_epochs=20, batch_size=50, lr=0.1,
                   display_step=1):
        """
        Finetuing the network
        """
        accu=[]
        accuu=[]
        print("\nStart finetuning...\n")
        start_time = timeit.default_timer()
        train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self.cost)
        batch_num = int(train_x.shape[0] / batch_size)
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("logs", sess.graph)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            for step in range(batch_num - 1):
                x_batch = train_x[step * batch_size:(step + 1) * batch_size]
                y_batch = train_y[step * batch_size:(step + 1) * batch_size]
                # 训练
                sess.run(train_op, feed_dict={self.x: x_batch, self.y: y_batch})
                # 计算cost
                avg_cost += sess.run(self.cost, feed_dict={self.x: x_batch, self.y:y_batch})/ batch_num
                # 输出
            if epoch % display_step == 0:
                val_acc = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                # accu.append(val_acc)
                # accuu.append(avg_cost)

                print("\tEpoch {0} cost: {1} accuracy:{2}".format(epoch, avg_cost,val_acc))

            # result = sess.run(merged, feed_dict={self.x: test_x, self.y: test_y})  # 输出
            # writer.add_summary(result, epoch)
        end_time = timeit.default_timer()
        print("\nThe finetuning process ran for {0} minutes".format((end_time - start_time) / 60))
        # y_aix = np.array(accu)
        # y_aix1=np.array(accuu)
        # x_aix = np.transpose(np.arange(1, 6))
        # plt.plot(x_aix, y_aix,label="predict")
        # plt.plot(x_aix,y_aix1,label="real")
        # plt.savefig("E:\\高若涵计算机毕设\\DBN_predict_performance\\picture\\test_p30_f3.jpg")
        # plt.show()

    def predict(self, sess, x_test=None):
        print("\nStart predict...\n")

        # predict_model = theano.function(
        #     inputs=[self.params],
        #     outputs=self.output_layer.y_pre)
        dbn_y_pre_temp = sess.run(self.output_layer.output, feed_dict={self.x: x_test})
        # print(dbn_y_pre_temp)
        dbn_y_pre = pd.DataFrame(mm.inverse_transform(dbn_y_pre_temp))
        dbn_y_pre.to_csv('NSW_06.csv')
        print("\nPredict over...\n")

    # SVR输出预测结果
    def svr_output(self,sess,test_x,test_y):
        input_svr=sess.run(self.layers[-1].output,feed_dict={self.x:test_x})
        print("\nsvr predict...\n")
        svr=SVR(gamma=0.0005,kernel='rbf',C=15,epsilon=0.008)
        rmse=np.sqrt(-cross_val_score(svr,input_svr,test_y.ravel(),scoring="neg_mean_squared_error",cv=5))
        score=cross_val_predict(svr,input_svr,test_y.ravel(),cv=5)
        print(rmse.mean())
        print(pd.DataFrame(mm.inverse_transform(score.reshape(-1,1))))

    # 网格搜索最优参数
    def grid_get(self,sess,model,test_x, test_y, param_grid):
        input_svr = sess.run(self.layers[-1].output, feed_dict={self.x: test_x})
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(input_svr, test_y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])


def loadDataSet(data, ratio):
    trainingData = []
    testData = []
    x=np.array(data).shape[0]

    for i in range(0,x):
        if random.random() < ratio:  # 数据集分割比例
            trainingData.append(data[i])  # 训练数据集列表
        else:
            testData.append(data[i])  # 测试数据集列表
    return trainingData, testData

def splitDataSet(filename):
    numFeat = len(open(filename).readline().split())-1
    dataMat= [];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr = []
        curline =line.strip().split()
        for i in range(numFeat):
            lineArr.append(float(curline[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curline[-1]))
    return np.array(dataMat),np.array(labelMat)


if __name__ == "__main__":

    graph = tf.Graph()
    with graph.as_default():
        input_size = 13  # 你输入的数据特征数量
        lr = 0.01
        train_ecpho = 50  # 训练次数

        start_time0 = timeit.default_timer()

        filename = 'Boston House Price Dataset.txt'
        dataMat, labelMat = splitDataSet(filename)
        # print(dataMat[0:3])
        # print(labelMat[0:3])
        trainX= dataMat[:300, :]
        trainY = labelMat[:300]
        testX = dataMat[300:, :]
        testY = labelMat[300:]

        x_train = MaxMinNormalization(trainX)
        # print(x_train[0:3])
        y_train = MaxMinNormalization(np.transpose([trainY]))
        # print(y_train[0:3])
        x_test = MaxMinNormalization(testX)

        y_test = MaxMinNormalization(np.transpose([testY]))

        sess = tf.Session(graph=graph)
        dbn = DBN(n_in=x_train.shape[1], n_out=1, hidden_layers_sizes=[13,10,10])
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        tf.set_random_seed(seed=1111)

        dbn.pretrain(sess, x_train, lr=lr, pretraining_epochs=100)
        dbn.finetuning(sess, x_train, y_train, x_test, y_test, lr=lr, training_epochs=100)
        # dbn.grid_get(sess,model=SVR(),test_x=x_test,test_y=y_test,param_grid={'C': [9, 11, 13, 15], 'kernel': ["rbf"], "gamma": [0.0003, 0.0004,0.0005], "epsilon": [0.008, 0.009]})
        dbn.svr_output(sess,x_test,y_test)
        # dbn.predict(sess, x_test)

        file_name = 'saved_model/model.ckpt'  # 将保存到当前目录下的的saved_model文件夹下model.ckpt文件
        saver.save(sess, file_name)


        end_time0 = timeit.default_timer()
        print("\nThe Predict process ran for {0}".format((end_time0 - start_time0)))

