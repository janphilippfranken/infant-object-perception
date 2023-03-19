import os
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from tqdm import tqdm
from data import ObjectData
from vqvae import VQVAE


def train_vqvae(world):
    root_dir = f"../../data/worlds/unlabeled/{world}"
    dataset = ObjectData(root_dir=root_dir, is_tensor=False)
    epochs = 500
    num_embeddings = 5
    embedding_dim = 8
    device = torch.device("cpu")

    model = VQVAE(in_channels=3, 
                  num_hiddens=64, 
                  num_residual_layers=2, 
                  num_residual_hiddens=32,
                  num_embeddings=num_embeddings,
                  embedding_dim=embedding_dim,
                  commitment_cost=2.0,
                  ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    x_hats, quants, recon_errors, perplexities, min_encodings = [], [], [], [], []

    for epoch in tqdm(range(epochs)):
        model.train()
        x_in = dataset.data.to(device)

        z, vq_loss, x_hat, perplexity, quant, min_encoding, min_encoding_indices, distance = model(x_in)
        recon_error = F.mse_loss(x_hat, x_in) 
        loss = recon_error + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        x_hats.append(x_hat)
        quants.append(quant)
        recon_errors.append(recon_error.item())
        perplexities.append(perplexity.item())
        min_encodings.append(min_encoding_indices)

    os.makedirs(f"../codebooks/{world}", exist_ok=True)
    for img, code in enumerate(min_encodings[-1].view(10, 64, 64)):
        torch.save(code, f"../codebooks/{world}/codes_{img}.pt")

def main():
    worlds = [str(i) for i in range(1,9)]
    Parallel(n_jobs=8)(delayed(train_vqvae)(world) for world in worlds)

if __name__ == "__main__":
    main()