import pandas as pd
import numpy as np

from icd_analysis_helper import ICDAnalysisHelper
import matplotlib.pyplot as plt
import wordcloud
import collections


class PatientCluster:
    def __init__(self, 
                patient_icd_df_with_cluster_assignments, 
                cluster):
        self.cluster = cluster
        self.cluster_df = patient_icd_df_with_cluster_assignments[patient_icd_df_with_cluster_assignments.CLUSTER == cluster]
        
        self.disease_df = self.cluster_df.drop(['SUBJECT_ID', 'CLUSTER'], axis=1)
        self.disease_distribution = self.disease_df.sum(axis=0).sort_values(ascending=False)
        
    def get_cluster_ids(self):
        return self.cluster_df.SUBJECT_ID

    def plot_disease_distribution(self, top_k_diseases: int=10, use_short_titles: bool=True):
        
        disease_distribution = self.disease_distribution[:top_k_diseases].sort_values(ascending=True)
        plt.figure(figsize=(5, 10))
        plt.barh(np.arange(disease_distribution.size),
           disease_distribution.to_numpy(),
           align='center')
        
        if use_short_titles:
            title_keys = map(lambda icd9code: ICDAnalysisHelper.get_icd_short_titles([icd9code]), 
                                disease_distribution.keys() )
        else:
            title_keys = map(lambda icd9code: ICDAnalysisHelper.get_icd_long_titles([icd9code]), 
                                disease_distribution.keys() )
        plt.yticks(np.arange(disease_distribution.size), title_keys)
        plt.xlim(0, np.max(disease_distribution))
        plt.show()

    def generate_wordcloud(self,
                           use_short_titles:bool=False,
                           weight_by_disease_frequency:bool=True,
                           max_words_in_cloud=200):
        ###Titles
        disease_freqs = self.disease_distribution.values
        
        if use_short_titles:
            titles = map(lambda icd9code: ICDAnalysisHelper.get_icd_short_titles([icd9code]),self.disease_distribution.index )
        else:
            titles = map(lambda icd9code: ICDAnalysisHelper.get_icd_long_titles([icd9code]),self.disease_distribution.index )
        
        ###Title Word Frequencies
        text_processor = WordCloud()
        all_word_freqs = collections.Counter()
        for i, title in enumerate(titles):
            title_word_counts = text_processor.process_text(title)
            
            #Reset all word counts to 1. This is in case the long title has multiple occurences of the same word
            title_word_counts.update((word, 1) for word in title_word_counts.keys())
            
            #Weight word by number of occurences in cluster
            if weight_by_disease_frequency:
                title_word_counts.update((word, disease_freqs[i]*count) for word, count in title_word_counts.items())
            
            all_word_freqs.update(title_word_counts)
                
        cloud = WordCloud(max_words=max_words_in_cloud, 
                          background_color="white",
                          colormap="cool").generate_from_frequencies(all_word_freqs)
        
        plt.figure(figsize=(10, 10))
        plt.title("WordCloud for Patient Cluster {}".format(self.cluster))
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        