from collections import Counter
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
import nltk
nltk.download('stopwords')

class ClusterProcessor():
    def __init__(self, patient_icd_binary, icd9codes, icd9diag, cluster_assignments, k_neighbors, other_stopwords = []):
        self.patient_icd_binary = patient_icd_binary
        self.icd9codes = icd9codes
        self.icd9diag = icd9diag
        self.cluster_assignments = cluster_assignments
        self.k_neighbors = k_neighbors
        self.other_stopwords = ['unspecified', 'without', "elsewhere", 'type', 'mention', 
                              'chronic', 'acute', 'failure', 'coronary', 'due', 'essential', 
                              'use', 'heart', 'kidney', 'disease', 'diseases', 'specified', 
                              'classified', 'complication', 'history']
        self.other_stopwords += other_stopwords


    def get_cluster_wordcloud(self, c, mc_cluster=False, plot=True):
        if not mc_cluster:
            patient_idx = self.cluster_assignments[self.cluster_assignments["CLUSTER"] == c]["ORIGINAL_INDEX"].values
        else:
            patient_idx = self.cluster_assignments[self.cluster_assignments["MC_CLUSTER"] == c]["ORIGINAL_INDEX"].values
        size = len(patient_idx)
        remaining_patient_icd_binary = self.patient_icd_binary[patient_idx]

        titles = []         
        for i in range(remaining_patient_icd_binary.shape[0]):
            patient_icd_idx = np.nonzero(remaining_patient_icd_binary[i])[1]
            patient_icd_codes = self.icd9codes.iloc[patient_icd_idx].values.flatten()
            diaglongtitles=self.icd9diag[self.icd9diag["ICD9_CODE"].isin(patient_icd_codes)]["LONG_TITLE"].values.tolist()
            titles += diaglongtitles
        
        med_stopwords = set(stopwords.words('english'))
        med_stopwords.update(self.other_stopwords)

        text_processor = WordCloud(stopwords=med_stopwords)
        all_word_freqs = Counter()
        for i, title in enumerate(titles):
            title = title.lower()
            title_word_counts = text_processor.process_text(title)
            all_word_freqs.update(title_word_counts)
        
        if plot:
            cloud = WordCloud(max_words=30, 
                              background_color="white",
                              colormap="cool").generate_from_frequencies(all_word_freqs)
            plt.figure(figsize=(10, 10))
            plt.title("WordCloud for Patient Cluster {}".format(c))
            plt.imshow(cloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig("./figures/wordclouds/cluster{}K{}size{}.png".format(c, self.k_neighbors, size))
            plt.show()

        return all_word_freqs

    def get_cluster_labels(self, mc_cluster=False):
        if mc_cluster:
            cluster_labels = self.cluster_assignments['MC_CLUSTER'].values
        else:
            cluster_labels = self.cluster_assignments['CLUSTER'].values
        return cluster_labels 

    def build_cluster_matrix(self, mc_cluster=False):
        cluster_labels = self.get_cluster_labels(mc_cluster)
        knn_clusters = np.unique(cluster_labels).shape[0]
        K = np.zeros((cluster_labels.shape[0], knn_clusters))
        for i, c in enumerate(cluster_labels):
            K[i, c] = 1
        return csr_matrix(K)

    def sort_by_cluster_membership(self, M, mc_cluster=False):
        cluster_labels = self.get_cluster_labels(mc_cluster)
        sorted_cluster_labels = sorted(list(enumerate(cluster_labels.tolist())), key=lambda p: p[1])
        clusters_by_idx = [p[0] for p in sorted_cluster_labels]
        return M[clusters_by_idx][:, clusters_by_idx]

    def compute_centroids(self, y, mc_cluster=False):

        cluster_labels = self.get_cluster_labels(mc_cluster)
        cluster_memberships = {}
        for idx, c in enumerate(cluster_labels):
            if c not in cluster_memberships:
                cluster_memberships[c] = [idx]
            else:
                cluster_memberships[c].append(idx)
        
        unique_labels = np.unique(cluster_labels)
        centroids = np.zeros((unique_labels.shape[0], y.shape[-1]))
        for c in tqdm(unique_labels):
            cluster_c_idx = cluster_memberships[c]
            cluster_c_embeddings = y[cluster_c_idx]
            # centroids.append(np.mean(cluster_c_embeddings, axis=0))
            centroids[c] = np.mean(cluster_c_embeddings, axis=0)
            if cluster_c_embeddings.shape[0] == 0:
                print("Empty cluster found:", c)
        return np.array(centroids)

    def top_diseases_in_cluster(self, cluster, topk=3):
        remaining_patient_idxs = self.cluster_assignments[self.cluster_assignments["CLUSTER"] == cluster]["ORIGINAL_INDEX"].values
        remaining_patient_icd_binary = self.patient_icd_binary[remaining_patient_idxs]
        disease_distribution = np.sum(remaining_patient_icd_binary, axis=0).tolist()[0]

        icdidx_count_topk = sorted(list(enumerate(disease_distribution)), key=lambda p: p[1], reverse=True)[:topk]
        
        icdidx = [p[0] for p in icdidx_count_topk]
        disease_counts_topk = [p[1] for p in icdidx_count_topk]
        
        icd9codes_topk = [self.icd9codes.iloc[idx]["ICD9_CODE"] for idx in icdidx]
        # print(icd9codes_topk)
        # print(self.icd9diag[self.icd9diag['ICD9_CODE'] == '9974']["LONG_TITLE"].values)

        titles = []
        for icd9code in icd9codes_topk:
            title = self.icd9diag[self.icd9diag['ICD9_CODE'] == icd9code]["LONG_TITLE"].values
            if len(title) == 0:
                titles.append(icd9code)
            else:
                titles.append(title[0])
        # titles = list(map(lambda icd9code: self.icd9diag[self.icd9diag['ICD9_CODE'] == icd9code]["LONG_TITLE"].values[0], icd9codes_topk))

        return pd.DataFrame({"ICD9_CODE": icd9codes_topk,
                             "LONG_TITLE": titles,
                             "DISEASE_COUNT": disease_counts_topk})
        # return list(zip(icd9codes_topk, titles, disease_counts_topk))

    def plot_disease_distribution(self, topk, cluster=None, plot=True):
        if cluster == None:
            remaining_patient_idxs = self.cluster_assignments["ORIGINAL_INDEX"].values
            remaining_patient_icd_binary = self.patient_icd_binary[remaining_patient_idxs]
            disease_distribution = np.sum(remaining_patient_icd_binary, axis=0).tolist()[0]
        else:
            remaining_patient_idxs = self.cluster_assignments[self.cluster_assignments["CLUSTER"] == cluster]["ORIGINAL_INDEX"].values
            remaining_patient_icd_binary = self.patient_icd_binary[remaining_patient_idxs]
            disease_distribution = np.sum(remaining_patient_icd_binary, axis=0).tolist()[0]

        icdidx_count_topk = sorted(list(enumerate(disease_distribution)), key=lambda p: p[1], reverse=True)[:topk]
        
        icdidx = [p[0] for p in icdidx_count_topk]
        disease_counts_topk = [p[1] for p in icdidx_count_topk]
        
        icd9codes_topk = [self.icd9codes.iloc[idx]["ICD9_CODE"] for idx in icdidx]

        titles = []

        for icd9code in icd9codes_topk:
            title = self.icd9diag[self.icd9diag['ICD9_CODE'] == icd9code]["LONG_TITLE"].values
            if len(title) == 0:
                titles.append(icd9code)
            else:
                titles.append(title[0])
        # titles = list(map(lambda icd9code: self.icd9diag[self.icd9diag['ICD9_CODE'] == icd9code]["LONG_TITLE"].values[0], icd9codes_topk))

        if plot:
            plt.figure(figsize=(5, 10))
            plt.barh(np.arange(len(disease_counts_topk)),
                     disease_counts_topk,
                     align='center')

            plt.yticks(np.arange(len(disease_counts_topk)), titles)
            plt.xlim(0, np.max(disease_counts_topk))
            plt.gca().invert_yaxis()
            if cluster == None:
                plt.title("Disease Distribution: Top {}".format(topk))
            else:
                plt.title("Disease Distribution: Cluster {}, Top {}".format(cluster, topk))
            plt.show()
        return list(zip(icd9codes_topk, titles, disease_counts_topk))