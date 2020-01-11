from typing import List, Tuple
from tsnecuda import TSNE
from umap import UMAP

import numpy as np 
import pandas as pd

import torch
from torch import nn, optim

import matplotlib.pyplot as plt

class Visualizer:
    def tsne_embedding(X):
        return TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)

    def umap_embedding(X):
        return UMAP().fit_transform(X)

    def plot2d(X, filename, colors=None):
        plt.figure(figsize=(8,5))
        plt.scatter(
            x=X[:, 0], 
            y=X[:, 1],
            c=colors, 
            cmap='cool', 
            alpha=0.05
        )
        plt.xlabel('tsne-one')
        plt.ylabel('tsne-two')
        plt.colorbar()
        
        plt.savefig("{}.png".format(filename))
        plt.show()

