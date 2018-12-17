"""
Deep Belief Network
author: Cuson
2018/12/16
"""
import timeit
import os
from BP import BPNeuralNetwork
import tensorflow as tf
import numpy as np
from mlp import HiddenLayer
from rbm import RBM
import random
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
            # We use the LogisticRegression layer as the output layer
            with tf.name_scope('output_layer'):
                self.output_layer = BPNeuralNetwork(inpt=self.layers[-1].output, n_in=hidden_layers_sizes[-1],n_out=n_out)
        self.params.extend(self.output_layer.params)
        # The finetuning cost
        with tf.name_scope('output_loss'):
            self.cost = self.output_layer.cost(self.y)
        # The accuracy
        # self.accuracy = self.output_layer.accuarcy(self.y)

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
                sess.run(train_op, feed_dict={self.x: x_batch, self.y: np.transpose([y_batch])})
                # 计算cost
                avg_cost += sess.run(self.cost, feed_dict={self.x: x_batch, self.y:np.transpose([y_batch])})/ batch_num
                # 输出
            if epoch % display_step == 0:
                val_acc = sess.run(self.cost, feed_dict={self.x: test_x, self.y: np.transpose([test_y])})
                print("\tEpoch {0} cost: {1} accuracy:{2}".format(epoch, avg_cost,val_acc))

            # result = sess.run(merged, feed_dict={self.x: test_x, self.y: test_y})  # 输出
            # writer.add_summary(result, epoch)
        end_time = timeit.default_timer()
        print("\nThe finetuning process ran for {0} minutes".format((end_time - start_time) / 60))

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
    # keep_prob = 0.6  # dropout的概率

    graph = tf.Graph()
    with graph.as_default():
        input_size = 13  # 你输入的数据特征数量
        lr = 0.001
        train_ecpho = 50  # 训练次数

        filename = 'Boston House Price Dataset.txt'
        dataMat,labelMat=splitDataSet(filename)
        # print(dataMat[0:3])
        # print(labelMat[0:3])
        xs = tf.placeholder(dtype=tf.float32, shape=[None, 13])
        mean, std = tf.nn.moments(xs, axes=[0])
        scale = 0.1
        shift = 0
        epsilon = 0.001
        data = tf.nn.batch_normalization(xs, mean, std, shift, scale, epsilon)
        dbn = DBN(n_in=13, n_out=1, hidden_layers_sizes=[900,900,500])
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
       sess.run(mean, feed_dict={xs: dataMat})
       sess.run(std, feed_dict={xs: dataMat})
       num = sess.run(data, feed_dict={xs:dataMat})
       # print(num)


       trainX=num[:400,:]
       trainY =labelMat[:400]
       testX=dataMat[400:,:]
       testY = labelMat[400:]

       # print(trainX[0:3])
       # print(trainY[0:3])
       # print(testX[0:3])
       # print(testY[0:3])

       sess.run(init)

       tf.set_random_seed(seed=99999)
       np.random.seed(123)
       dbn.pretrain(sess, trainX, pretraining_epochs=train_ecpho,lr=lr)
       dbn.finetuning(sess, trainX, trainY, testX, testY, lr=lr, training_epochs=train_ecpho)

       file_name = 'saved_model/model.ckpt'  # 将保存到当前目录下的的saved_model文件夹下model.ckpt文件
       saver.save(sess, file_name)

