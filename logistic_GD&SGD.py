'''
author:滕依筱 918106840206
description:logistic_GD&SGD
write time:2020-10-26 11:11
last modified:2020-11-19 21:53
######修改：封装了第二次作业的类&修改了输出的图片######
通过梯度下降法和随机梯度下降法求解逻辑回归
（因为牛顿法没有用均值归一法所以将文件单独分开）
'''
import numpy as np
import matplotlib.pyplot as plt

#logistic
class Logistic:
    def read_data(self):
        #读取x数据
        x = np.array([[]])
        y = np.array([[]])
        theta = np.array([[10], [0.1], [0.1]])
        with open("ex4x.dat", 'r') as f:
            data = f.readlines()  # 将txt中所有字符串读入data
            for line in data:
                numbers = line.split()  # 将数据分隔
                numbers_float = list(map(float, numbers))  # 转化为浮点数
                x = np.append(x, 1)
                x = np.append(x, numbers_float[0])
                x = np.append(x, numbers_float[1])
        x = x.reshape(80, 3).transpose()

        #对x使用均值归一法
        meanx1 = np.mean(x[1:])
        meanx2 = np.mean(x[2:])
        min1 = np.amin(np.array(x[1:]))
        min2 = np.amin(np.array(x[2:]))
        max1 = np.amax(np.array(x[1:]))
        max2 = np.amax(np.array(x[2:]))
        for i in range(0, 79):
            x[1, i] = (x[1, i] - meanx1) / (max1 - min1)
            x[2, i] = (x[2, i] - meanx2) / (max2 - min2)

        #读取y数据
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

    #梯度下降法迭代过程
    def GD_run(self, x, y, theta):
        alpha = 0.05
        index = []
        cost = []
        for i in range(0, 1000):
            h = self.H(x, theta)
            sum0 = sum1 = sum2 = 0
            for k in range(0, y.size - 1):
                sum0 += (h[0, k] - y[k])
                sum1 += (h[0, k] - y[k]) * x[1, k]
                sum2 += (h[0, k] - y[k]) * x[2, k]
            theta[0, 0] -= (alpha / y.size) * sum0
            theta[1, 0] -= (alpha / y.size) * sum1
            theta[2, 0] -= (alpha / y.size) * sum2
            index.append(i)
            cost.append(self.Cost(x, y, theta))
        plt.subplot(2, 2, 1)
        plt.title("Gradient Descent")
        plt.plot(index, cost)
        print("Gradient Descent: j=", self.Cost(x, y, theta))
        print("Gradient Descent: final theta_0:", theta[0][0], ",theta_1:", theta[1][0], ",theta_2:", theta[2][0])
        return theta

    #梯度下降法画图
    def GD_draw(self, theta):
        xx = np.arange(-0.5, 0.3, 0.001)
        yy = (theta[1, 0] * xx + theta[0, 0]) / (-theta[2, 0])
        plt.subplot(2, 1, 2)
        plt.plot(xx, yy, color='red')

    #随机梯度下降法迭代过程
    def SGD_run(self, x, y, theta):
        alpha = 0.05
        index = []
        jj = []
        for i in range(0, 1000):
            number = np.random.randint(0, 79)
            xx = theta[0, 0] * x[0, number] + theta[1, 0] * x[1, number] + theta[2, 0] * x[2, number]
            temp0 = ((1 / (1 + np.exp(-xx))) - y[number])
            temp1 = x[1, number] * ((1 / (1 + np.exp(xx))) - y[number])
            temp2 = x[2, number] * ((1 / (1 + np.exp(xx))) - y[number])
            theta[0, 0] -= alpha * temp0
            theta[1, 0] -= alpha * temp1
            theta[2, 0] -= alpha * temp2
            index.append(i)
            jj.append(self.Cost(x, y, theta))
        plt.subplot(2, 2, 2)
        plt.title("Stochastic Gradient Descent")
        plt.plot(index, jj)
        print("Stochastic Gradient Descent: j=", self.Cost(x, y, theta))
        print("Stochastic Gradient Descent: final theta_0:", theta[0][0], ",theta_1:", theta[1][0], ",theta_2:",
              theta[2][0])
        return theta

    #随机梯度下降法画图
    def SGD_draw(self, theta):
        xx = np.arange(-0.5, 0.3, 0.001)
        yy = (theta[1, 0] * xx + theta[0, 0]) / (-theta[2, 0])
        plt.subplot(2, 1, 2)
        plt.plot(xx, yy, color='green')

    #所有的图片显示
    def all_draw(self, x, y):
        plt.legend(["Gradient Descent", "Stochastic Gradient Descent"])
        for i in range(0, 79):
            if y[i] == 0:
                plt.scatter(x[1, i], x[2, i], c='red')
            else:
                plt.scatter(x[1, i], x[2, i], c='blue')
        plt.show()

    #logistic回归（使用GD和SGD）
    def run(self):
        x, y, theta = self.read_data()
        theta1 = self.GD_run(x, y, theta)
        self.GD_draw(theta)
        x, y, theta = self.read_data()
        theta = self.SGD_run(x, y, theta1)
        self.SGD_draw(theta)
        self.all_draw(x,y)

if __name__ == "__main__":
    logistic = Logistic()
    logistic.run()
'''
最终的图片中红线为梯度下降法，绿线为随机梯度下降法
此处可以看出，梯度下降法相较随机梯度下降法收敛较快。
（梯度下降法需要约400次迭代，随机梯度下降法则需要约900次）
而且随机梯度下降法在收敛过程中有较明显的波动，
但是随机梯度下降法单次迭代所需要的时间复杂度较小，
总体上一次梯度下降法等于80次随机梯度下降法的工作量。
二者最终损失值计算结果都比较接近0.4（牛顿法结果）。
'''
