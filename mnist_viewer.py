# Created by Boao Yang.
import numpy as np
import matplotlib.pyplot as plt

def view(dataset, idx):
    image = dataset.images[idx] # shape = 784 (28*28)
    label = dataset.labels[idx]
    plt.imshow(np.reshape(image,[28,28]))
    plt.title('MNIST number: ' + str(label))
    plt.axis('off')
    plt.show()