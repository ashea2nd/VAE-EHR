import numpy as np 
import pandas as pd

import torch
from torch import nn, optim

from vae import VAE
from vae import bce_kld_loss_function, train

patient_icd_file = "/Users/andrew/Documents/meng/spring/PATIENT_ICD.csv"
patient_icd_df = pd.read_csv(patient_icd_file, sep=' ')

patient_icd_data = patient_icd_df.drop('SUBJECT_ID', axis=1)
data = torch.tensor(patient_icd_data.values)
print(patient_icd_data)
print(data)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_dim = data.shape[1]
print("Feature_dim: {}".format(feature_dim))
encoder_dim = [(100, 200), (200, 100), (100, 50)]
latent_dim = 25
decoder_dim = [(50, 100), (100, 200)]
model = VAE(
    feature_dim = feature_dim, 
    encoder_dim = encoder_dim,
    latent_dim = latent_dim,
    decoder_dim = decoder_dim
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)




