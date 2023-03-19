import torch
import torch.nn as nn

# model
from encoder import Encoder
from vq import VQ
from decoder import Decoder


class VQVAE(nn.Module):
    
    def __init__(self, 
                 in_channels,
                 num_hiddens, 
                 num_residual_layers, 
                 num_residual_hiddens, 
                 num_embeddings, 
                 embedding_dim, 
                 commitment_cost, 
                ):
        
        super(VQVAE, self).__init__()
        
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        # self.decay = decay
        
        # encoder 
        self.encoder = Encoder(in_channels=self.in_channels, 
                               num_hiddens=self.num_hiddens, 
                               num_residual_layers=self.num_residual_layers, 
                               num_residual_hiddens=self.num_residual_hiddens,
                              )
        
        # pre-vq conv num_hiddens -> embedding_dim
        self.pre_vq_conv = nn.Conv2d(in_channels=self.num_hiddens,
                                     out_channels=self.embedding_dim,
                                     kernel_size=1,
                                    )
        
        # vq
        self.vq = VQ(num_embeddings=self.num_embeddings, 
                     embedding_dim=self.embedding_dim,
                     commitment_cost=self.commitment_cost,
                    )
    
        # decoder 
        self.decoder = Decoder(in_channels=self.embedding_dim, 
                               num_hiddens=self.num_hiddens,
                               num_residual_layers=self.num_residual_layers,
                               num_residual_hiddens=self.num_residual_hiddens,
                              )
        
        
    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        loss, quantized, perplexity, min_encodings, min_encoding_indices, distances = self.vq(z)
        x_hat = self.decoder(quantized)
        
        return z, loss, x_hat, perplexity, quantized, min_encodings, min_encoding_indices, distances
    