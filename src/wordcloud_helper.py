from collections import Counter
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

class WordCloudProcessor():
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


    def get_cluster_wordcloud(self, c, mc_cluster=False):
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
        
        cloud = WordCloud(max_words=30, 
                          background_color="white",
                          colormap="cool").generate_from_frequencies(all_word_freqs)
            
        plt.figure(figsize=(10, 10))
        plt.title("WordCloud for Patient Cluster {}".format(c))
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig("./figures/wordclouds/cluster{}K{}size{}.png".format(c, self.k_neighbors, size))
        plt.show()

    def plot_disease_distribution(self, topk, cluster=None):
        if cluster == None:
            remaining_patient_idxs = self.cluster_assignments["ORIGINAL_INDEX"].values
            remaining_patient_icd_binary = self.patient_icd_binary[remaining_patient_idxs]
            disease_distribution = np.sum(remaining_patient_icd_binary, axis=0).tolist()[0]
        else:
            remaining_patient_idxs = self.cluster_assignments[self.cluster_assignments["CLUSTER"] == cluster]["ORIGINAL_INDEX"].values
            remaining_patient_icd_binary = self.patient_icd_binary[remaining_patient_idxs]
            disease_distribution = np.sum(remaining_patient_icd_binary, axis=0).tolist()[0]

        icdidx_count_topk = sorted(list(enumerate(disease_distribution)), key=lambda p: p[1])[-topk:]
        
        icdidx = [p[0] for p in icdidx_count_topk]
        disease_counts_topk = [p[1] for p in icdidx_count_topk]
        
        icd9codes_topk = [self.icd9codes.iloc[idx]["ICD9_CODE"] for idx in icdidx]
        titles = map(lambda icd9code: self.icd9diag[self.icd9diag['ICD9_CODE'] == icd9code]["LONG_TITLE"].values[0], icd9codes_topk)

        plt.figure(figsize=(5, 10))
        plt.barh(np.arange(len(disease_counts_topk)),
                 disease_counts_topk,
                 align='center')

        plt.yticks(np.arange(len(disease_counts_topk)), titles)
        plt.xlim(0, np.max(disease_counts_topk))
        if cluster == None:
            plt.title("Disease Distribution: Top {}".format(topk))
        else:
            plt.title("Disease Distribution: Cluster {}, Top {}".format(cluster, topk))
        plt.show()
        return icd9codes_topk
    
