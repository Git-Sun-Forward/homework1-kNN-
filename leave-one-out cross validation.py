import os, sys
import numpy as np
import matplotlib.pyplot as plt

import mnist_dataloader
import mnist_viewer
import test as t
import time

mnist_dataset = mnist_dataloader.read_data_sets("./MNIST_dataset/")
# generate 3 datasets: A (train), B (test), C (A & B mixed)
dataset_A, dataset_B, dataset_C = mnist_dataset.train, mnist_dataset.test, mnist_dataset.multi

# you can use index to get specific item (e.g. image_A[0])
images_A, images_B, images_C = dataset_A.images, dataset_B.images, dataset_C.images
labels_A, labels_B, labels_C = dataset_A.labels, dataset_B.labels, dataset_C.labels

matrix_C=t.images_to_matrix(dataset_C)
start=time.perf_counter()
D=t.euclidean_distance_matrix(matrix_C,matrix_C)
np.fill_diagonal(D,np.inf)
nn_idx=np.argmin(D,axis=1)
pred=labels_C[nn_idx]
time_dist=time.perf_counter() - start
accuracy=np.mean(pred == labels_C)
print('LOOCV accuracy:{}, total time:{}'.format(accuracy,time_dist))