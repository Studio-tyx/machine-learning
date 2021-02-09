'''
author:滕依筱 918106840206
description:svm compared with Logistic and ANN
(Logistic based on SGD & ANN based on TensorFlow)
time:2020-11-12 21:36
通过调用LibSVM库实现向量机，并与Logistics（SGD）回归、ANN（TensorFlow）进行对比
'''
from libsvm323.python.commonutil import *
from libsvm323.python.svmutil import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.random import RandomState
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def readData():
    """
    输入函数：读取文件中的数值并且做归一化处理
    :return: X：train set, Y：label
    """
    # 读取"ex4x.dat"中x的值
    X = np.array([[]])
    Y = []
    with open("ex4x.dat", 'r') as f:
        data = f.readlines()
        for line in data:
            numbers = line.split()  # 将数据分隔
            numbers_float = list(map(float, numbers))  # 转化为浮点数
            X = np.append(X, [numbers_float[0], numbers_float[1]])
    X = X.reshape(80, 2)

    # 归一化处理
    meanx1 = np.mean(X[1:])
    meanx2 = np.mean(X[2:])
    min1 = np.amin(np.array(X[1:]))
    min2 = np.amin(np.array(X[2:]))
    max1 = np.amax(np.array(X[1:]))
    max2 = np.amax(np.array(X[2:]))

    # 不可以在for里面直接写(X[i,1] - np.mean(X[1:])) / (np.amax(np.array(X[1:])) - np.amin(np.array(X[1:]))) 因为会一直改变均值
    for i in range(0, 80):
        X[i, 1] = (X[i, 1] - meanx1) / (max1 - min1)
        X[i, 0] = (X[i, 0] - meanx2) / (max2 - min2)

    # 读取"ex4y.dat"中y的值
    with open("ex4y.dat", 'r') as f:
        data = f.readlines()
        for line in data:
            numbers = line.split()  # 将数据分隔
            numbers_float = map(float, numbers)  # 转化为浮点数
            for i in numbers_float:
                Y.append([int(i)])
    return X, Y


class Logistic:
    """
    Logistics Regression
    """

    def sigmoid(self, x, theta):
        """
        sigmoid函数
        :param x:
        :param theta:
        :return: 1/(1+np.exp(theta.T*x))
        """
        h = np.dot(theta.transpose(), x)
        for i in range(0, h.size):
            h[0, i] = 1 / (1 + np.exp(-(h[0, i])))
        return h

    def Cost(self, x, y, theta):
        """
        损失函数
        :param x:
        :param y:
        :param theta:
        :return: 交叉熵（为防止log0等情况 设置epsilon=1e-10）
        """
        j = 0
        h = self.sigmoid(x, theta)
        epsilon = 1e-10
        for i in range(0, y.size - 1):
            tmp = h[0, i]
            if y[i] == 0:
                j += np.log(1 - tmp + epsilon)
            else:
                j += np.log(tmp + epsilon)
        j /= (-y.size)
        return j

    def SGD(self, x, y, theta):
        """
        随机梯度下降法
        :param x:
        :param y:
        :param theta:
        :return: 经过迭代学习后的theta
        """
        alpha = 0.05
        # index = []
        # jj = []
        for i in range(0, 1000):
            number = np.random.randint(0, 79)
            xx = theta[0, 0] * x[0, number] + theta[1, 0] * x[1, number] + theta[2, 0] * x[2, number]
            temp0 = ((1 / (1 + np.exp(-xx))) - y[number])
            temp1 = x[1, number] * ((1 / (1 + np.exp(xx))) - y[number])
            temp2 = x[2, number] * ((1 / (1 + np.exp(xx))) - y[number])
            theta[0, 0] -= alpha * temp0
            theta[1, 0] -= alpha * temp1
            theta[2, 0] -= alpha * temp2
            # index.append(i)
            # jj.append(self.Cost(x, y, theta))
        print("Stochastic Gradient Descent: j=", self.Cost(x, y, theta))
        print("Stochastic Gradient Descent: final theta_0:", theta[0][0], ",theta_1:", theta[1][0], ",theta_2:",
              theta[2][0])
        return theta

    @staticmethod
    def draw(x, y, theta):
        """
        画图
        :param x:
        :param y:
        :param theta:
        :return:
        """

        xx = np.arange(-0.5, 0.3, 0.001)
        plt.subplot(2, 2, 2)
        plt.title("Logistic")
        yy = (theta[1, 0] * xx + theta[0, 0]) / (-theta[2, 0])
        plt.plot(xx, yy, color='green')
        for i in range(0, 79):
            if y[i] == 0:
                plt.scatter(x[1, i], x[2, i], c='red')
            else:
                plt.scatter(x[1, i], x[2, i], c='blue')

    def run(self):
        """
        Logistic总运行函数
        :return:
        """
        X, Y = readData()
        X = np.insert(X.transpose(), 0, np.ones([1, 80]), axis=0)
        Y = np.array(Y)
        theta = np.array([[10], [0.1], [0.1]])
        theta = self.SGD(X, Y, theta)
        self.draw(X, Y, theta)


class ANN:

    @staticmethod
    def train(X, Y):
        """
        训练函数（通过TensorFlow）
        :param X:
        :param Y:
        :return:
        """

        # 神经网络初始化参数
        w = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))
        b = tf.Variable(tf.random.normal([1], stddev=1, seed=1))
        x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
        y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
        y = tf.nn.sigmoid(tf.matmul(x, w) + b)
        cross_entropy = -tf.reduce_mean(
            y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
        train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

        # TensorFlow建立会话
        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
            print("--------------wait for ANN training---------------")
            # 迭代次数为300
            steps = 300
            for i in range(steps):
                for (input_x, input_y) in zip(X, Y):
                    input_x = np.reshape(input_x, (1, 2))
                    input_y = np.reshape(input_y, (1, 1))
                    sess.run(train_step, feed_dict={x: input_x, y_: input_y})

                # 每迭代100次输出一次日志信息
                if i % 100 == 0:
                    # 输出交叉熵之和
                    total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
                    print("ANN:After", i, "training steps, cost is", total_cross_entropy)

            # 生成网络采样点
            N, M = 200, 200
            t1 = np.linspace(min(X[:, 0]), max(X[:, 0]), N)
            t2 = np.linspace(min(X[:, 1]), max(X[:, 1]), M)
            x1, x2 = np.meshgrid(t1, t2)
            x_show = np.stack((x1.flat, x2.flat), axis=1)

            # 预测
            y_predict = sess.run(y, feed_dict={x: x_show})

            # 作图
            plt.subplot(2, 2, 1)
            plt.title("ANN")
            cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0F0'])
            plt.pcolormesh(x1, x2, np.array(y_predict).reshape(x1.shape), cmap=cm_light)
            plt.scatter(X[:, 0], X[:, 1], c=Y, s=3, marker='o')
            # plt.show()

    def run(self):
        X, Y = readData()
        self.train(X, Y)


class SVM:
    @staticmethod
    def linear_train():
        """
        用线性模型训练
        :return:使用linear_train得到的训练模型
        """

        # 训练模型
        y, x = svm_read_problem('data1.txt')  # 读取数据（由原始数据经过格式转换后的文件）
        m = svm_train(y[:80], x[:80], '-t 0 -e 1')  # 训练模型（根据线性核函数）1为软间隔大小
        svm_predict(y, x, m)  # 原样本的分析结果
        return m

    @staticmethod
    def high_dimension_train():
        """
        训练模型
        :return:使用svm_train得到的训练模型
        """

        # 训练模型
        y, x = svm_read_problem('data1.txt')  # 读取数据（由原始数据经过格式转换后的文件）
        m = svm_train(y[:80], x[:80], '-c 4')  # 训练模型（规定惩罚值的4较小 因此核函数向高维扩展）
        svm_predict(y, x, m)  # 原样本的分析结果
        return m

    @staticmethod
    def draw(m, flag):
        """
        根据已得到的训练模型画出分类图
        :param flag: 作图依据（线性核函数在左下角显示图形 高维核函数在右下角）
        :param m:已得到的训练模型
        :return:
        """

        # 再次读取数据
        # 主要是为了得到x1与x2未归一化的最大最小值（LibSVM中读取问题时没有归一化）
        # 以便建立网状采样点
        X1 = []
        X2 = []
        Y1 = []
        with open("ex4x.dat", 'r') as f:
            data = f.readlines()  # 将txt中所有字符串读入data
            for line in data:
                numbers = line.split()  # 将数据分隔
                numbers_float = list(map(float, numbers))  # 转化为浮点数
                X1 = np.append(X1, numbers_float[0])
                X2 = np.append(X2, numbers_float[1])
        with open("ex4y.dat", 'r') as f:
            data = f.readlines()  # 将txt中所有字符串读入data
            for line in data:
                numbers = line.split()  # 将数据分隔
                numbers_float = map(float, numbers)  # 转化为浮点数
                for i in numbers_float:
                    Y1 = np.append(Y1, i)
        x1_np = np.asarray(X1)
        x2_np = np.asarray(X2)
        y_np = np.asarray(Y1)

        # 生成网络采样点
        N, M = 200, 200
        x1_min = x1_np.min(axis=0)
        x2_min = x2_np.min(axis=0)
        x1_max = x1_np.max(axis=0)
        x2_max = x2_np.max(axis=0)
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)
        x_show = np.stack((x1.flat, x2.flat), axis=1)

        # 预测
        y_fake = np.zeros((40000,))
        y_predict, _, _ = svm_predict(y_fake, x_show, m)

        # 作图
        if flag:
            plt.subplot(2, 2, 3)  # linear_kernel
            plt.title("svm(linear_kernel)")
        else:
            plt.subplot(2, 2, 4)  # high_dimension_kernel
            plt.title("svm(high_dimension_kernel)")
        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0F0'])
        plt.pcolormesh(x1, x2, np.array(y_predict).reshape(x1.shape), cmap=cm_light)
        plt.scatter(x1_np, x2_np, c=y_np, s=3, marker='o')
        # plt.show()

    def run(self):
        """
        svm的总运行函数
        分别使用线性核函数和高维函数学习
        :return:
        """
        print("---------------linear_train result-----------------")
        train_model = self.linear_train()
        self.draw(train_model, True)
        print("------------high_dimension_train result------------")
        train_model = self.high_dimension_train()
        self.draw(train_model, False)


if __name__ == "__main__":
    logistic=Logistic()
    logistic.run()

    ann=ANN()
    ann.run()

    svm=SVM()
    svm.run()

    plt.show()

"""
本次作业中使用LibSVM建立了svm模型
并与之前写的Logistic回归、基于TensorFlow的ANN神经网络进行对比
在svm模型中通过线性核函数和高维核函数对于数据进行学习
高维核函数是通过设置较小的惩罚值实现的
由结果可以发现高维核函数学习结果中数据准确度较高
但是有一些过拟合了，因此还是线性核函数效果更好
与Logistic回归、ANN对比发现，ANN所需时间较大，而通过SGD实现的Logistic准确度较低
"""
