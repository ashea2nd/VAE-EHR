from typing import List, Tuple
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm
from torch.distributions import Normal, Poisson, kl_divergence as kl


class VAE(nn.Module):

    def __init__(
        self,
        feature_dim: int, 
        encoder_dim: List[Tuple[int, int]],
        latent_dim: int,
        decoder_dim: List[Tuple[int, int]],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_relu: bool = True,
        bias: bool = True
        ):
        super().__init__()

        #Encoder
        encoder_dim.insert(0, (feature_dim, encoder_dim[0][0]))
        self.encoder = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in, n_out, bias=bias),
                            nn.BatchNorm1d(n_out) if use_batch_norm else None,
                            nn.ReLU() if use_relu else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(encoder_dim)
                ]
            )
        )

        self.mean_encoder = nn.Linear(encoder_dim[-1][-1], latent_dim)
        self.var_encoder = nn.Linear(encoder_dim[-1][-1], latent_dim)

        #Decoder
        if len(decoder_dim) == 0:
            self.decoder = []
            self.output_decoder = nn.Linear(latent_dim, feature_dim)
        else:
            decoder_dim.insert(0, (latent_dim, decoder_dim[0][0]))
            self.decoder = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            "Layer {}".format(i),
                            nn.Sequential(
                                nn.Linear(n_in, n_out, bias=bias),
                                nn.BatchNorm1d(n_out) if use_batch_norm else None,
                                nn.ReLU() if use_relu else None,
                                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                            ),
                        )
                    for i, (n_in, n_out) in enumerate(decoder_dim)
                    ]
                )
            )
            self.output_decoder = nn.Linear(decoder_dim[-1][-1], feature_dim)

        self.output_decoder_sigmoid = nn.Sigmoid()

    def reparameterize_gaussian(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def get_latent(self, x: torch.Tensor):
        q = self.encode(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = self.reparameterize_gaussian(q_m, q_v)
        return latent, q_m, q_v

    def encode(self, x: torch.Tensor):
        for layers in self.encoder:
            for layer in layers:
                if layer is not None:
                    x = layer(x)
        return x

    def decode(self, x: torch.Tensor):
        for layers in self.decoder:
            for layer in layers:
                if layer is not None:
                    x = layer(x)
        return x

    def forward(self, x: torch.Tensor):
        #Pass thru Encoder
        q = self.encode(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = self.reparameterize_gaussian(q_m, q_v)

        #Pass thru Decoder
        y = self.decode(latent)
        y = self.output_decoder(y)
        y = self.output_decoder_sigmoid(y)

        #q_v = torch.clamp(q_v, max=100.)
        #Saw that variance blows up around order 1e2

        #print("Variance min max:", torch.min(q_v), torch.max(q_v))
        #print("Mean min max:", torch.min(q_v), torch.max(q_v))


        return y, q_m, q_v

class VAETrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        experiment_name: str,
        kld_beta: float = 1.0
        ):

        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.optimizer = optimizer

        self.experiment_name = experiment_name

        self.elbos_per_epoch = []
        self.ave_kld_per_epoch = []
        self.ave_bce_per_epoch = []

        self.kld_beta = kld_beta

    def bce_kld_loss_function(
        self, 
        recon_x: torch.Tensor, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
        ):
        #view() explanation: https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch
        #print("Min, max in recon:", torch.min(recon_x), torch.max(recon_x))
        #print("Min, max in orig:", torch.min(x), torch.max(x))

        BCE = F.binary_cross_entropy(recon_x, x.view(-1, recon_x.shape[1]), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -self.kld_beta*0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD, BCE, KLD

    def train(
        self,
        data: torch.Tensor,
        epochs: int = 800, 
        batch_size: int = 20,
        kld_beta: float = 1.0,
        save_model_interval: int = 50,
        log_interval: int = 100,
        clip_gradients: bool = False,
        grad_norm_limit: float = 5
        ):

        self.elbos_per_epoch = []
        self.bce_per_epoch = []
        self.kld_per_epoch = []
        print("Training with KLD Beta weight of {}".format(self.kld_beta))
        for epoch in range(1, epochs+1):
            self.model.train()
            train_loss = 0
            data_length = data.shape[0]
            
            assert data_length % batch_size == 0, "data and batch size are not compatible. Data Size: {}, Batch Size: {}".format(data_length, batch_size)
            
            train_bce = 0
            train_kld = 0
            for i in tqdm(range(0, data_length, batch_size)):
                batch = data[i:i + batch_size]
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(batch)
                loss, bce, kld = self.bce_kld_loss_function(recon_batch, batch, mu, logvar)
                loss.backward()

                train_loss += loss.item()
                train_bce += bce.item()
                train_kld += kld.item()

                #print("GRAD NORMS", [p.grad.data.norm(2) for p in model.parameters()])
                #Gradients are either super large or super small, ranging from 0 to 1e13 before blowing up
                # clip_grad_norm(self.model.parameters(), 5)
                if clip_gradients:
                    clip_grad_norm(self.model.parameters(), grad_norm_limit) #A value of 5 was shown to work

                self.optimizer.step()

                # batch_idx = i / batch_size
                # if batch_idx % log_interval == 0:
                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(batch), len(data),
                #         100. * batch_idx / (data_length/batch_size),
                #         loss.item() / len(data)))
            
            total_batches = data_length / batch_size
            elbo_for_epoch = train_loss/total_batches
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, elbo_for_epoch))
            bce_for_epoch = train_bce / total_batches
            kld_for_epoch = train_kld / total_batches

            self.elbos_per_epoch.append(elbo_for_epoch)
            self.bce_per_epoch.append(bce_for_epoch)
            self.kld_per_epoch.append(kld_for_epoch)

            if epoch % save_model_interval == 0:
                torch.save(self.model.state_dict(), "VAE_exp_{}_epoch_{}.pkl".format(self.experiment_name, epoch))

        torch.cuda.empty_cache()

    def encode_data(self, data: torch.Tensor):
        self.model.eval()
        data = data.to(self.device)
        return self.model.get_latent(data)

    def reconstruct_data(self, data: torch.Tensor):
        torch.cuda.empty_cache()
        self.model.eval()
        data = data.to(self.device)
        recon_x, q_m, q_v = self.model(data)
        loss, BCE, KLD = self.bce_kld_loss_function(recon_x=recon_x, x=data, mu=q_m, logvar=q_v)
        return latent, q_m, q_v, loss, BCE, KLD

    def plot_elbo(self):
        plt.figure(figsize=(8,5))
        plt.plot(np.log(self.elbos_per_epoch))
        plt.ylabel("Log ELBO")
        plt.xlabel("Epoch")
        plt.savefig("ELBO_{}.png".format(self.experiment_name))
        plt.show()

    def plot_bce(self):
        plt.figure(figsize=(8,5))
        plt.plot(np.log(self.bce_per_epoch))
        plt.ylabel("Log BCE")
        plt.xlabel("Epoch")
        plt.savefig("BCE_{}.png".format(self.experiment_name))
        plt.show()

    def plot_kld(self):
        plt.figure(figsize=(8,5))
        plt.plot(np.log(self.kld_per_epoch))
        plt.ylabel("Log KLD")
        plt.xlabel("Epoch")
        plt.savefig("KLD_{}.png".format(self.experiment_name))
        plt.show()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# feature_dim = 6985
# encoder_dim = [(100, 200), (200, 100), (100, 50)]
# latent_dim = 25
# decoder_dim = [(50, 100), (100, 200)]
# model = VAE(
#     feature_dim = feature_dim, 
#     encoder_dim = encoder_dim,
#     latent_dim = latent_dim,
#     decoder_dim = decoder_dim
# ).to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)


# print(model)














