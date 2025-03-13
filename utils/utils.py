import csv
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata


# 设置随机数种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 绘制神经网络训练的收敛过程
def plot_l2(filename, output_filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        x = []
        y = []
        for row in reader:
            x.append(int(row[0]))
            y.append(float(row[1]))

    plt.plot(x, y)
    plt.xlim(left=0, right=max(x))
    plt.ylim(bottom=0.9 * min(y), top=1.1 * max(y))
    plt.gca().set_yscale('log')
    # plt.xlabel('迭代次数')
    # plt.ylabel('MSE损失')
    plt.savefig(output_filename, dpi=300)
    plt.show()


def plot_eq(filename, output_filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        x = []
        y = []
        for row in reader:
            x.append(int(row[0]))
            y.append(float(row[2]))

    plt.plot(x, y)
    plt.xlim(left=0, right=max(x))
    plt.ylim(bottom=0.9 * min(y), top=1.1 * max(y))
    plt.gca().set_yscale('log')
    # plt.xlabel('迭代次数')
    # plt.ylabel('PDE残差')
    plt.savefig(output_filename, dpi=300)
    plt.show()


def plot_uv(filename, output_filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        x = []
        u = []
        v = []
        for row in reader:
            x.append(int(row[0]))
            u.append(float(row[1]))
            v.append(float(row[2]))

    plt.plot(x, u)
    plt.plot(x, v)
    plt.xlim(left=0, right=max(x))
    plt.ylim(bottom=0.9 * min(min(u), min(v)), top=1.1 * max(max(u), max(v)))
    plt.gca().set_yscale('log')
    plt.xlabel('迭代次数')
    plt.ylabel('相对误差')
    plt.savefig(output_filename, dpi=300)
    plt.show()


# 绘制神经网络预测出的解的二维等值线图
def plot_solution(X_star, u_star, index, filename):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    plt.figure(index, figsize=(5, 3))
    plt.pcolor(X, Y, U_star, cmap='jet')
    plt.colorbar()
    plt.savefig(filename, dpi=300)
    plt.show()

# 调整3D轴的比例使其保持一致
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
