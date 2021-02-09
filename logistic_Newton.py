'''
author:滕依筱 918106840206
description:logistic_newton
time:2020-10-26 11:39
last modified:2020-11-19 22:04
######修改：封装了第二次作业的类&修改了输出的图片######
通过牛顿法求解逻辑回归
'''
import numpy as np
import matplotlib.pyplot as plt


# 逻辑回归（牛顿法）
class Logistic:
    # 读取数据
    def read_data(self):
        x = np.array([[]])
        y = np.array([[]])
        theta = np.array([[-10], [0.1], [0.1]])
        # 读取x数据
        with open("ex4x.dat", 'r') as f:
            data = f.readlines()  # 将txt中所有字符串读入data
            for line in data:
                numbers = line.split()  # 将数据分隔
                numbers_float = list(map(float, numbers))  # 转化为浮点数
                x = np.append(x, 1)
                x = np.append(x, numbers_float[0])
                x = np.append(x, numbers_float[1])
        x = x.reshape(80, 3).transpose()
        # 读取y数据
        with open("ex4y.dat", 'r') as f:
            data = f.readlines()  # 将txt中所有字符串读入data
            for line in data:
                numbers = line.split()  # 将数据分隔
                numbers_float = map(float, numbers)  # 转化为浮点数
                for i in numbers_float:
                    y = np.append(y, i)
        return x, y, theta

    # sigmoid函数
    @staticmethod
    def H(x, theta):
        h = np.dot(theta.transpose(), x)
        for i in range(0, h.size):
            h[0, i] = 1 / (1 + np.exp(-(h[0, i])))
        return h

    # 损失函数
    def Cost(self, x, y, theta):
        j = 0
        h = self.H(x, theta)
        epsilon = 1e-10
        for i in range(0, y.size - 1):
            tmp = h[0, i]
            if y[i] == 0:
                j += np.log(1 - tmp + epsilon)
            else:
                j += np.log(tmp + epsilon)
        j /= (-y.size)
        return j

    # 牛顿法迭代过程
    def NT(self, x, y, theta):
        index = []
        cost = []
        for i in range(0, 10):
            h = self.H(x, theta)
            J1 = (1 / y.size) * np.dot(x, (h - y).transpose())
            J2 = np.mat(
                (1.0 / y.size) * np.dot(x,
                                        np.dot(np.diag(np.matrix(np.multiply(h, (1 - h))).getA()[0]), x.transpose())))
            delta = np.linalg.pinv(J2) * J1
            theta = theta - delta
            index.append(i)
            cost.append(self.Cost(x, y, theta))
            print("Newton: j=", self.Cost(x, y, theta))
        plt.subplot(1, 2, 1)
        plt.title("Cost Decrease")
        plt.plot(index, cost)
        print("Newton: j=", self.Cost(x, y, theta))
        print("Newton: final theta_0:", theta[0][0], ",theta_1:", theta[1][0], ",theta_2:", theta[2][0])
        return theta

    # 牛顿法画图
    @staticmethod
    def draw(x, y, theta):
        xx = np.arange(0, 100)
        yy = (theta[1, 0] * xx + theta[0, 0]) / (-theta[2, 0])
        plt.subplot(1, 2, 2)
        plt.title("Newton")
        plt.plot(xx, yy, color='red')
        for i in range(0, 79):
            if y[i] == 0:
                plt.scatter(x[1, i], x[2, i], c='red')
            else:
                plt.scatter(x[1, i], x[2, i], c='blue')
        plt.show()

    # 运行
    def NT_run(self):
        x, y, theta = self.read_data()
        theta = self.NT(x, y, theta)
        self.draw(x, y, theta)


if __name__ == "__main__":
    logistic = Logistic()
    logistic.NT_run()

'''
这里可以看出，相较于梯度下降法，牛顿法所需要的迭代次数较少，且得到的损失值更小，数据更加精确。
我在将牛顿法的数据均值归一化后发现部分矩阵无法求逆，因此并没有将牛顿法的数据均值归一化。
（诚然我能力有限emmm也可能是其他原因导致的这个bug）
除了逆矩阵的问题，牛顿法每次计算的工作量也非常大，这二者是牛顿法的主要缺点。
'''
