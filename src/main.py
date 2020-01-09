import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn, optim

import vae
from vae import VAE, VAETrainer

patient_icd_file = "../../PATIENT_ICD_BINARY.csv"
patient_icd_df = pd.read_csv(patient_icd_file, sep=' ')

patient_icd_data = patient_icd_df.drop('SUBJECT_ID', axis=1)

data = torch.tensor(patient_icd_data.values).float()
print(data.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print((device))

feature_dim = data.shape[1]
print("Feature_dim: {}".format(feature_dim))
encoder_dim = [(250, 500), (500, 250), (250, 100)]
latent_dim = 10
decoder_dim = [(50, 100)]
model = VAE(
    feature_dim = feature_dim, 
    encoder_dim = encoder_dim,
    latent_dim = latent_dim,
    decoder_dim = decoder_dim
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
print(type(optimizer))

date="200108"
experiment_name = "{}_patient_clusters_linear_architecture".format(date)
trainer = VAETrainer(experiment_name=experiment_name)

trainer.train(
    model=model,
    device=device,
    optimizer=optimizer,
    data=data, 
    epochs=500,
    batch_size=40,
    save_model_interval=10
)