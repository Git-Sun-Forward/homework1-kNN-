import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt

import mnist_dataloader
import mnist_viewer


def images_to_matrix(dataset):
    """将数据集中的所有展平图片组合成一个 (N, 784) 的矩阵。

    每张 MNIST 图片是 28x28 = 784 个像素，加载器已经把每张图片展平为
    长度 784 的一维向量（dataset.images 形状为 (N, 784)）。本函数把这些
    行堆叠成一个矩阵返回，例如 dataset_A -> (3000, 784)。
    """
    return np.asarray(dataset.images, dtype=np.float32)


mnist_dataset = mnist_dataloader.read_data_sets("./MNIST_dataset/")
# generate 3 datasets: A (train), B (test), C (A & B mixed)
dataset_A, dataset_B, dataset_C = mnist_dataset.train, mnist_dataset.test, mnist_dataset.multi

train_size = dataset_A.num_examples
test_size = dataset_B.num_examples
print('Dataset size:', '(train, test) =', (train_size, test_size))

# 将数据集 A B 的所有展平图片组合成一个矩阵
matrix_A = images_to_matrix(dataset_A)
matrix_B = images_to_matrix(dataset_B)
# 计算距离
def euclidean_distance_matrix(A, B):
    # A: (m, d), B: (n, d)
    A2 = np.sum(A ** 2, axis=1, keepdims=True)        # (m, 1)
    B2 = np.sum(B ** 2, axis=1, keepdims=True).T      # (1, n)
    D2 = A2 + B2 - 2 * A @ B.T                        # (m, n)
    D2 = np.maximum(D2, 0)                            # 防负数
    return np.sqrt(D2)


def manhattan_distance_matrix(A, B):
    """曼哈顿距离 (L1): D[i,j] = sum_k |A[i,k] - B[j,k]|"""
    m, n = A.shape[0], B.shape[0]
    D = np.zeros((m, n), dtype=A.dtype)
    for k in range(A.shape[1]):
        D += np.abs(A[:, k, None] - B[None, :, k])
    return D


def chebyshev_distance_matrix(A, B):
    """L∞ 范数距离 (切比雪夫距离): D[i,j] = max_k |A[i,k] - B[j,k]|"""
    m, n = A.shape[0], B.shape[0]
    D = np.zeros((m, n), dtype=A.dtype)
    for k in range(A.shape[1]):
        D = np.maximum(D, np.abs(A[:, k, None] - B[None, :, k]))
    return D


def minkowski_distance_matrix(A, B, p=4):
    """闵可夫斯基距离: D[i,j] = (sum_k |A[i,k] - B[j,k]|^p)^(1/p)"""
    m, n = A.shape[0], B.shape[0]
    D = np.zeros((m, n), dtype=A.dtype)
    for k in range(A.shape[1]):
        D += np.abs(A[:, k, None] - B[None, :, k]) ** p
    return D ** (1.0 / p)

def nearest_neighbor_predict(dist_matrix, train_labels):
    """最近邻分类：对距离矩阵的每一行，返回最小值对应的训练标签。

    约定：dist_matrix[i, j] 表示第 i 个「测试样本」与第 j 个「训练样本」之间的距离，
    形状为 (n_test, n_train)。对每一行取 argmin，得到该测试样本最近邻的训练样本
    列索引，再映射到对应的训练标签。
    """
    nn_idx = np.argmin(dist_matrix, axis=1)   # 每行最小值对应的列索引
    return np.asarray(train_labels)[nn_idx]   # 映射为对应的标签


def knn_predict(dist_matrix, train_labels, k):
    """k 近邻分类：对距离矩阵的每一行，取距离最小的 k 个邻居，用多数投票决定标签。

    参数:
        dist_matrix: 形状 (n_test, n_train)，dist_matrix[i, j] = 第 i 个测试样本
                     与第 j 个训练样本之间的距离。
        train_labels: 训练集标签，长度 n_train。
        k:           邻居数量（由调用者手动设置）。

    返回:
        pred: 形状 (n_test,)，每个测试样本的预测标签。
    """
    train_labels = np.asarray(train_labels)
    # 每行距离最小的 k 个邻居的列索引（argpartition 只保证前 k 个是最小的，
    # 不保证内部有序，但多数投票不需要顺序，因此比 argsort 更快）
    knn_idx = np.argpartition(dist_matrix, k - 1, axis=1)[:, :k]
    knn_labels = train_labels[knn_idx]  # (n_test, k)

    n_classes = int(train_labels.max()) + 1
    pred = np.empty(knn_labels.shape[0], dtype=train_labels.dtype)
    for i in range(knn_labels.shape[0]):
        votes = np.bincount(knn_labels[i], minlength=n_classes)
        pred[i] = np.argmax(votes)  # 平票时取类别编号较小者
    return pred


# 创建 kNN 数组，自动循环分类
kNN = [3, 5, 7, 11, 21]

# 进行分类
labels_A = np.asarray(dataset_A.labels)
labels_B = np.asarray(dataset_B.labels)

# 计时（循环外）：距离矩阵对所有 k 都相同，只需计算一次
start = time.perf_counter()
distance = manhattan_distance_matrix(matrix_A, matrix_B)
time_dist = time.perf_counter() - start
print(f'Distance computation time: {time_dist:.4f} s')

# distance 形状为 (train, test)，转置后每一行是一个测试样本 -> (test, train)
# 计时（循环内）：对每个 k 单独计时预测，total = 距离时间 + 预测时间
for k in kNN:
    start = time.perf_counter()
    pred_labels = knn_predict(distance.T, labels_A, k)
    time_predict = time.perf_counter() - start
    total_time = time_dist + time_predict

    accuracy = np.mean(pred_labels == labels_B)
    print(f'k={k}: accuracy={accuracy:.4f}, prediction time={time_predict:.4f} s, '
          f'total time={total_time:.4f} s')
