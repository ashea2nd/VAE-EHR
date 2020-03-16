import sys
sys.path.append("./../../src")
from visualizer_helper import Visualizer
from cluster_helper import ClusterProcessor

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.sparse import csr_matrix
import scipy
import pickle
import os
from annoy import AnnoyIndex
from scipy.sparse.linalg import inv

from collections import Counter
from wordcloud import WordCloud

k_neighbors = 50
knn_clusters = 100
keep_k_evecs = 10
drop_k_evecs=2

cluster_assignments = pd.read_csv("./data/CLUSTER_ASSIGNMENTS.csv")
icd9codes = pd.read_csv("../../data/PATIENT_ICD_ICD9_CODES.csv")
patient_icd_binary = pickle.load(open("../../data/PATIENT_ICD_BINARY_SPARSE_CSR.p", 'rb'))
icd9diag = pd.read_csv("../../../mimic/D_ICD_DIAGNOSES.csv.gz")
icd9proc = pd.read_csv("../../../mimic/D_ICD_PROCEDURES.csv.gz")

A = pickle.load(open("./data/A_mknn_K{}_CSR.p".format(k_neighbors), "rb"))

Y_cluster_labels = pickle.load(open("./data/cluster_labels_K{}_knn{}_topkevecs{}_drop{}.p".format(k_neighbors, knn_clusters, keep_k_evecs, drop_k_evecs), 'rb'))
Y_mc_cluster_labels = pickle.load(open("./data/cluster_labels_mc_K{}_knn{}_topkevecs{}_drop{}.p".format(k_neighbors, knn_clusters, keep_k_evecs, drop_k_evecs), 'rb'))

cp = ClusterProcessor(patient_icd_binary, icd9codes, icd9diag, cluster_assignments, k_neighbors,
                        other_stopwords = ["hypertension", 'disorder'])

lc=100

Y_umap_2d = pickle.load(open("./data/Y_umap_2d_K{}_topkevecs{}_drop{}_lc{}.p".format(k_neighbors, keep_k_evecs, drop_k_evecs, lc), 'rb'))
Y_umap_3d = pickle.load(open("./data/Y_umap_3d_K{}_topkevecs{}_drop{}_lc{}.p".format(k_neighbors, keep_k_evecs, drop_k_evecs, lc), 'rb'))

Y_mc_umap_2d = pickle.load(open("./data/Y_umap_2d_mc_K{}_topkevecs{}_drop{}_lc{}.p".format(k_neighbors, keep_k_evecs, drop_k_evecs, lc), 'rb'))
Y_mc_umap_3d = pickle.load(open("./data/Y_umap_3d_mc_K{}_topkevecs{}_drop{}_lc{}.p".format(k_neighbors, keep_k_evecs, drop_k_evecs, lc), 'rb'))

Y_c_umap_2d = cp.compute_centroids(Y_umap_2d, mc_cluster=False)
Y_c_umap_3d = cp.compute_centroids(Y_umap_3d, mc_cluster=False)
Y_mc_c_umap_2d = cp.compute_centroids(Y_mc_umap_2d, mc_cluster=True)
Y_mc_c_umap_3d = cp.compute_centroids(Y_mc_umap_3d, mc_cluster=True)

def create_centroid_cluster_df(y, cluster_labels, topk=3):
    centroids = cp.compute_centroids(y, mc_cluster=False)
    
    x = centroids[:, 0].tolist()
    y = centroids[:, 1].tolist()
    cluster_sizes = np.unique(cluster_labels, return_counts=True)[1].tolist()
        
    titles = ["" for _ in range(centroids.shape[0])]
    print(len(titles))
    for c in tqdm(np.unique(cluster_labels)):
        top_k_df = cp.top_diseases_in_cluster(cluster=c, topk=3)
        top_k_titles = top_k_df["LONG_TITLE"].values.tolist()
        top_k_titles_asstr = "\n".join(titles)
        titles[c] = top_k_titles_asstr
        
        
    centroid_dict = {"x": x,
                     "y": y,
                     "cluster_size": cluster_sizes,
                     "cluster": np.unique(cluster_labels).tolist(),
                     "title": titles
                    }
    
    if centroids.shape[-1] == 3:
        z = centroids[:, 2].tolist()
        centroid_dict['z'] = z
        
    return pd.DataFrame(centroid_dict)

print(np.unique(Y_cluster_labels))


Y_2d_df = create_centroid_cluster_df(Y_umap_2d, Y_cluster_labels)

plt.figure(figsize=(15,15))
ax = sns.scatterplot(x="x", y="y", size="cluster_size", hue="cluster",
            sizes=(300, 2000), alpha=.5, palette="muted",
            data=Y_2d_df)
for line in tqdm(range(0,Y_2d_df.shape[0])):
     ax.text(Y_2d_df.x[line], Y_2d_df.y[line], Y_2d_df.title[line])

h,l = ax.get_legend_handles_labels()
plt.legend(h[-5:], l[-5:], loc="upper right")

plt.savefig("./figures/CLUSTER_PLOT_TOP_WORDS.png")