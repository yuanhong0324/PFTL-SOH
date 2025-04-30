import math
import os
import random

import torch
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import seaborn as sns

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_metrix(true_label, pred_label):
    MAE = metrics.mean_absolute_error(true_label, pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label, pred_label)
    MSE = metrics.mean_squared_error(true_label, pred_label)
    RMSE = np.sqrt(metrics.mean_squared_error(true_label, pred_label))
    return [MAE, RMSE]


def save_to_txt(save_name, string):
    f = open(save_name, mode='a')
    f.write(string)
    f.write('\n')
    f.close()


def cosine_annealing_lr(base_lr, epoch, T_max):
    return base_lr * (1 + math.cos(math.pi * epoch / T_max)) / 2


def set_seed(seed):
    # 设置PyTorch随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # 设置Python随机种子
    random.seed(seed)
    # 设置NumPy随机种子
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("成功设置 Random Seed:", seed)


def plot_histograms(original_data, generated_data,save_dir):
    n_features = original_data.shape[1]
    plt.figure(figsize=(n_features, 9))

    for i in range(n_features):
        plt.subplot(2, n_features // 2 + 1, i + 1)
        plt.hist(original_data[:, i], bins=30, alpha=0.5, label='O')
        plt.hist(generated_data[:, i], bins=30, alpha=0.5, label='G')
        # plt.title(f'Feature {i + 1} Histogram')
        # plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'histograms.png'))


def plot_kde(original_data, generated_data,save_dir):
    n_features = original_data.shape[1]
    plt.figure(figsize=(n_features, 9))

    for i in range(n_features):
        plt.subplot(2, n_features//2 + 1, i + 1)
        sns.kdeplot(original_data[:, i], label='O', fill=True)
        sns.kdeplot(generated_data[:, i], label='G', fill=True)
        plt.title(f'F {i + 1} KDE')
        plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'kde.png'))


def plot_cdf(original_data, generated_data,save_dir):
    n_features = original_data.shape[1]
    plt.figure(figsize=(n_features, 9))

    for i in range(n_features):
        plt.subplot(2, n_features//2 + 1, i + 1)

        # 计算原始数据的CDF
        original_sorted = np.sort(original_data[:, i])
        original_cdf = np.arange(1, len(original_sorted) + 1) / len(original_sorted)
        plt.plot(original_sorted, original_cdf, label='O Data')

        # 计算生成数据的CDF
        generated_sorted = np.sort(generated_data[:, i])
        generated_cdf = np.arange(1, len(generated_sorted) + 1) / len(generated_sorted)
        plt.plot(generated_sorted, generated_cdf, label='G Data')

        plt.title(f'F {i + 1} CDF')
        plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'cdf.png'))
