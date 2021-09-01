import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy(x, accuracy, methods, title, xlabel='x', ylabel='accuracy'):
    for i in range(len(methods)):
        plt.plot(x, accuracy[i], label=methods[i])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()
