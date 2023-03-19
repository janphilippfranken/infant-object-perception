# VQ layer - quantises tensor; channel diemnsions will be used as the space in which to quantise -> BHWC - flatten all except C
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cpu")

class VQ(nn.Module):
    
    def __init__(self, 
                num_embeddings, 
                embedding_dim, 
                commitment_cost,
               ):
        super(VQ, self).__init__()
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim  
        self.commitment_cost = commitment_cost
        # genetic algorithm
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
    
    def forward(self,
                inputs,
               ):
        # permute BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs_shape = inputs.shape
        
        # B*H*W, C where C = self.embedding_dim
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # distances j||z_e(x) - e_j||^2
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t())) 
        
        # encoding zq(x) = ek, where k = argmin j||ze(x) − ej||^2
        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # vis the argmin stuff; plot bar plot over epochs
        
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        quantized = torch.matmul(min_encodings, self.embedding.weight).view(inputs_shape)
        
        # loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        loss = q_latent_loss + self.commitment_cost * e_latent_loss 

        quantized = inputs + (quantized - inputs).detach()
        
        avg_probs = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return loss, quantized, perplexity, min_encodings, min_encoding_indices, distances