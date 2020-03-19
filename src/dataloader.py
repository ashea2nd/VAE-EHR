import torch
from torch.utils.data import Dataset
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
import scipy
import pickle
import numpy as np
import annoy
from annoy import AnnoyIndex
import os
 

ANNOY_PATH = "./../../../large_data_files"

class MixehrICDImputationDataset(Dataset):

    def __init__(
        self, 
        # subject_ids_path: str, 
        patient_topic_distribution_path: str,
        icd_topic_feature_distribution_path: str
        ):
        self.patient_topic_distribution_df = pd.read_csv(patient_topic_distribution_path)
        self.icd_topic_feature_distribution_df = pd.read_csv(icd_topic_feature_distribution_path)

        self.icd_labels = self.icd_topic_feature_distribution_df["ICD9_CODE"].values

        topic_cols = [str(i) for i in range(1, 76)]
        self.patient_topic_data = self.patient_topic_distribution_df.to_numpy()
        self.icd_topic_distribution = self.icd_topic_feature_distribution_df[topic_cols].to_numpy()

    def get_feat_dim(self):
        return self.icd_topic_distribution.T.shape[1]

    def __len__(self):
        return len(self.patient_topic_distribution_df.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patient_topic_subset = self.patient_topic_data[idx]
        imputed_subset = patient_topic_subset @ self.icd_topic_distribution.T
        return imputed_subset

class PatientICDSparseVanillaDataset(Dataset):
    def __init__(
        self, 
        csr_data_path: str
        ):
        self.patient_data_csr = pickle.load(open(csr_data_path, 'rb'))
        print("Loaded CSR Dataset w/ dim {}".format(self.patient_data_csr.shape))
    def __len__(self):
        return self.patient_data_csr.shape[0]

    def __getitem__(self, idx):
        return self.get_patient_as_sparse_torch_tensor(idx)

    def get_feat_dim(self):
        return self.patient_data_csr.shape[1]

    def get_patient_as_sparse_torch_tensor(self, patient_idx):
        #Converts CSR matrix to COO matrix form
        csr_row = np.asarray(self.patient_data_csr[patient_idx].todense())[0]
        return csr_row.astype(np.float64)
        # coo = coo_matrix(csr_row)
        # idxs = np.vstack((coo.row, coo.col))

        # i = torch.LongTensor(idxs) 
        # vec = torch.FloatTensor(coo.data)
        # sparse_torch_tensor = torch.sparse.FloatTensor(i, vec, torch.Size(coo.shape))

        # return sparse_torch_tensor.to_dense()

class PatientEigenVectorDataset(Dataset):
    def __init__(
        self, 
        eigenvector_path: str
        ):
        self.evectors = pickle.load(open(eigenvector_path, 'rb'))
        print("Loaded eigenvector matrix of size {}".format(self.evectors.shape))

    def __len__(self):
        return self.evectors.shape[0]

    def __getitem__(self, idx):
        return self.evectors[idx]

    def get_feat_dim(self):
        return self.evectors.shape[1]

class PatientSparseSimilarityDataset(Dataset):
    def __init__(
        self, 
        csr_data_path: str,
        experiment_name: str,
        use_top_k_neighbors: int = 20, 
        matrix_rows_idxs = None
        ):

        self.use_top_k_neighbors = use_top_k_neighbors
        self.patient_data_csr = pickle.load(open(csr_data_path, 'rb'))
        if type(matrix_rows_idxs) != None:
            self.patient_data_csr = self.patient_data_csr[matrix_rows_idxs]

        print("Loaded CSR Dataset w/ dim {}".format(self.patient_data_csr.shape))

        self.knn_annoy_tree = self.build_annoy_index(filename=experiment_name)

    def __len__(self):
        return self.patient_data_csr.shape[0]

    def __getitem__(self, idx):
        sims = np.zeros((self.__len__()))
        nn_idxs = self.knn_annoy_tree.get_nns_by_item(idx, self.use_top_k_neighbors)
        distances = [self.knn_annoy_tree.get_distance(idx, nn) for nn in nn_idxs]
        sims[nn_idxs] = distances
        return sims

    def get_feat_dim(self):
        return self.patient_data_csr.shape[0]

    def build_annoy_index(
        self,
        filename: str,
        distance_metric: str = 'angular',
        n_trees: int = 10
        ):
        feature_size = self.patient_data_csr.shape[1]

        try:
            return self.load_annoy_index(filename, distance_metric)
        except OSError:
            pass
        print("Building ANNOY index...")
        if distance_metric in ['cosine', 'angular']:
            distance_metric = 'angular'

        knn_tree = AnnoyIndex(feature_size, distance_metric)
        for i in range(self.patient_data_csr.shape[0]):
            knn_tree.add_item(i, self.patient_data_csr[i].toarray()[0])    
        knn_tree.build(n_trees)
        knn_tree.save(os.path.join(ANNOY_PATH,"{}.ann".format(filename)))
        print("    Complete.")
        return knn_tree

    def load_annoy_index(
        self, 
        filename: str, 
        distance_metric: str="angular"
        ):
        feature_size = self.patient_data_csr.shape[-1]
        knn_tree = AnnoyIndex(feature_size, distance_metric)

        annoy_location = os.path.join(ANNOY_PATH, "{}.ann".format(filename))
        knn_tree.load(annoy_location)

        print("Loaded existing ANNOY index from {}".format(annoy_location))
        
        return knn_tree

class PatientSparseSimilarityDatasetMultipleVisits(PatientSparseSimilarityDataset):
    def __init__(
        self, 
        csr_data_path: str, 
        experiment_name: str,
        subject_ids_path: str,
        use_top_k_neighbors: int = 20
        ):

        self.subject_ids_df = pd.read_csv(subject_ids_path)
        matrix_row_idxs = self.subject_ids_df.MATRIX_ROW_IDX.to_numpy()

        super().__init__(csr_data_path, experiment_name, use_top_k_neighbors, matrix_row_idxs)


# class PatientDataSparseCSR():

#     def __init__(
#         self, 
#         subject_ids_path_from_config: str, 
#         csr_data_path_from_config: str
#         ):

#         self.subject_ids_df = pd.read_csv(subject_ids_path_from_config)
#         self.patient_data_csr = pickle.load(open(csr_data_path_from_config, 'rb'))
#         assert self.subject_ids_df.shape[0] == self.patient_data_csr.shape[0], "subject ID and data dimension mismatch. {} subject ids but {} rows in data matrix.".format(self.subject_ids_df.shape[0], self.patient_data_csr.shape[0])


#     def get_patient_as_sparse_torch_tensor(self, patient_idx):
#         #Converts CSR matrix to COO matrix form
#         csr_row = self.patient_data_csr[patient_idx]
#         coo = coo_matrix(csr_row)
#         idxs = np.vstack((coo.row, coo.col))

#         i = torch.LongTensor(idxs) 
#         vec = torch.FloatTensor(coo.data)
#         sparse_torch_tensor = torch.sparse.FloatTensor(i, vec, torch.Size(coo.shape)).to_dense()
        
#         return sparse_torch_tensor

# class PatientICDSparseDataset(PatientDataSparseCSR, Dataset):

#     def __init__(
#         self, 
#         subject_ids_path_from_config: str, 
#         csr_data_path_from_config: str
#         ):

#         super().__init__(subject_ids_path_from_config, csr_data_path_from_config)

#     def __len__(self):
#         return len(self.subject_ids_df.index)

#     def __getitem__(self, idx):

#         return self.get_patient_as_sparse_torch_tensor(idx)

# class PatientICDSparseSimilarityDataset(PatientDataSparseCSR, Dataset):

#     def __init__(
#         self, 
#         subject_ids_path_from_config: str, 
#         csr_data_path_from_config: str
#         ):

#         super().__init__(subject_ids_path_from_config, csr_data_path_from_config)

#     def __len__(self):
#         return len(self.subject_ids_df.index)

#     def __getitem__(self, idx):
#         #TODO
#         return


#     def build_annoy_index(self,
#         filename: str,
#         distance_metric: str = 'angular',
#         n_trees: int = 10
#         ):

#         feature_size = self.patient_data_csr.shape[-1]

#         try:
#             return load_annoy_index(feature_size, filename)
#         except OSError:
#             pass
        
#         if distance_metric in ['cosine', 'angular']:
#             distance_metric = 'angular'

#         knn_tree = AnnoyIndex(feature_size, distance_metric)
#         for i in range(self.patient_data_csr.shape[0]):
#             knn_tree.add_item(i, self.patient_data_csr[i].toarray()[0])    
#         knn_tree.build(n_trees)
#         knn_tree.save("{}.ann".format(filename))
#         return knn_tree

#     def load_annoy_index(self, 
#                          filename: str, 
#                          distance_metric: str="angular"):
#         feature_size = self.patient_data_csr.shape[-1]
#         knn_tree = AnnoyIndex(feature_size, distance_metric)
#         knn_tree.load("{}.ann".format(filename))
#         return knn_tree

#     def construct_knn_graph(batch: scipy.sparse, 
#                              full: scipy.sparse,
#                              filename: str = "PATIENT_ICD_ANNOY_INDEX_ANGULAR",
#                              distance_metric: str="cosine"):
        
#         feature_size = batch[0].shape[-1]
#         try:
#             annoy = load_annoy_index(feature_size, filename)
#         except OSError:
#             annoy = build_annoy_index(filename, distance_metric)
            
#         annoy = load_annoy_index(full.shape[-1], filename)
#         sims = np.zeros((batch.shape[0], full.shape[0]))
#         for i, row in enumerate(batch):
#             nn_idxs = annoy.get_nns_by_item(i, 100)
#             distances = [annoy.get_distance(i, nn) for nn in nn_idxs]
#             sims[i][nn_idxs] = distances
            
#         return sims

    # def compute_sparse_similarity(batch: scipy.sparse, 
    #                               full: scipy.sparse, 
    #                               metric: str="cosine"):
    #     assert batch.shape[-1] == full.shape[-1], "batch and data feature dimensions do not match"
        
    #     if metric == "cosine":
    #         batch = normalize(csr_matrix(batch))
    #         full = normalize(csr_matrix(full))
    #         return batch.dot(full.T)
        




