from typing import List, Tuple
from tqdm import tqdm
import collections

import torch
from torch import nn, optim
from torch.nn import functional as F
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

    def reparameterize_gaussian(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def get_latent(self, x: torch.Tensor):
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = self.reparameterize_gaussian(q_m, q_v)
        return latent, q_m, q_v

    def forward(self, x: torch.Tensor):
        #Pass thru Encoder
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = self.reparameterize_gaussian(q_m, q_v)

        #Pass thru Decoder
        y = self.decoder(latent)
        y = torch.sigmoid(self.output_decoder(y))

        return y, q_m, q_v

def bce_kld_loss_function(recon_x, x, mu, logvar):
    #view() explanation: https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, recon_x.shape[1]), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim,
    data: torch.Tensor,
    epochs: int = 800, 
    batch_size: int = 20, 
    log_interval: int = 100,
    save_model_interval: int = 50
    ):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        data_length = data.shape[0]
        
        assert data_length % batch_size == 0, "data and batch size are not compatible. Data Size: {}, Batch Size: {}".format(data_length, batch_size)
        
        for i in tqdm(range(0, data_length, batch_size)):
            batch = data[i:i + batch_size]
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = bce_kld_loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            batch_idx = i / batch_size
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(batch), len(data),
                    100. * batch_idx / (data_length/batch_size),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / (data_length/batch_size)))

        if epoch % save_model_interval == 0:
            torch.save(model.state_dict(), "VAE_epoch_{}.pkl".format(epoch))


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














