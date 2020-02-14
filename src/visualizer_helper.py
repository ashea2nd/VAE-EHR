from typing import List, Tuple
try:
    from cuml import TSNE, UMAP
except:
    pass

import numpy as np 
import pandas as pd

import torch
from torch import nn, optim

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def tsne_embedding(self, X, n_components: int=2):
        return TSNE(n_components=n_components, method="barnes_hut").fit_transform(X)

    def umap_embedding(self, X, n_components: int=2):
        return UMAP(n_components=n_components).fit_transform(X)

    def plot2d(self, X, filename, colors=None, alpha=0.05, xlim: Tuple=None, ylim: Tuple=None):
        plt.figure(figsize=(8,5))
        plt.scatter(
            x=X[:, 0], 
            y=X[:, 1],
            c=colors, 
            cmap='cool', 
            alpha=alpha
        )
        plt.xlabel('component-one')
        plt.ylabel('component-two')
        plt.colorbar()

        if xlim != None:
            plt.xlim(xlim[0], xlim[1])
        if ylim != None:
            plt.ylim(ylim[0], ylim[1])
        
        plt.savefig("{}.png".format(filename))
        plt.show()

    def plot3d(self, X, filename, colors=None, alpha=0.05, xlim: Tuple=None, ylim: Tuple=None):
        fig = plt.figure(figsize=(8,5))
        ax = Axes3D(fig)

        plot = ax.scatter(
            xs=X[:, 0], 
            ys=X[:, 1],
            zs=X[:, 2],
            c=colors, 
            cmap='cool', 
            alpha=alpha
        )
        ax.set_xlabel('component-one')
        ax.set_ylabel('component-two')
        ax.set_zlabel('component-three')
        fig.colorbar(plot)

        if xlim != None:
            plt.xlim(xlim[0], xlim[1])
        if ylim != None:
            plt.ylim(ylim[0], ylim[1])

        plt.savefig("{}.png".format(filename))
        plt.show()