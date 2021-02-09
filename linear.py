'''
author:滕依筱 918106840206
description:linear regression
time:2020-10-21 15:00
通过梯度下降法和正规方程法求解线性回归
x为年份
y为该年份的房价
y=theta0+theta1*x
'''

import numpy as np
import math
import matplotlib.pyplot as plt

'''
gradient descent
梯度下降法
'''
def GD(x, y, theta):
    delta = 1.0
    j = 0.0
    alpha = 0.00000001
    while (0.000001 < math.fabs(delta)):
        temp = (alpha / 14) * np.dot((np.dot(theta.transpose(), x) - y), x.transpose())
        theta -= temp.transpose()
        delta = j
        j = 0
        for i in range(14):
            j += (theta[0][0] + theta[1][0] * x[1][i] - y[0][i]) ** 2
        delta = delta - j
        # print("Gradient Descent: j=", j)
    print("Gradient Descent: j=", j)
    print("Gradient Descent: final theta_0:", theta[0][0], ",theta_1:", theta[1][0])
    return theta


'''
regular expression
正规方程法（最小二乘法）
'''
def RE(x, y):
    x_sum = x2_sum = y_sum = xy = 0
    for i in range(14):
        x_sum += x[1][i]
        y_sum += y[0][i]
        xy += x[1][i] * y[0][i]
        x2_sum += x[1][i] ** 2
    theta1 = (x_sum * y_sum / 14 - xy) / (x_sum ** 2 / 14 - x2_sum)
    theta0 = (y_sum - theta1 * x_sum) / 14
    j = 0
    for i in range(14):
        j += (theta0 + theta1 * x[1][i] - y[0][i]) ** 2
    print("Regular Expression: j=", j)
    print("Regular Expression: final theta_0:", theta0, ",theta_1:", theta1)
    return theta0, theta1


'''
构造[x0,x1]与[y]的矩阵
'''
x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]])
y = np.array([[2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900]])

'''
梯度下降法中以theta矩阵储存返回值
'''
theta = GD(x, y, np.array([[-1600.0], [0.0]]))
print("Gradient Descent: the price of 2014 is:", theta[0][0] + theta[1][0] * 2014)

'''
画图
'''
plt.subplot(1, 2, 1)
plt.scatter(x[1], y[0], c="r")
xx = np.arange(2000, 2015)
yy = theta[0][0] + theta[1][0] * xx
plt.plot(xx, yy)

'''
正规方程法中以theta0,theta1储存返回值
'''
theta0, theta1 = RE(x, y)
print("Regular Expression: the price of 2014 is:", theta0 + theta1 * 2014)

'''
画图
'''
yy = theta0 + theta1 * xx
plt.subplot(1, 2, 2)
plt.scatter(x[1], y[0], c="r")
plt.plot(xx, yy)
plt.show()
