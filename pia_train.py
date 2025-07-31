#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Physics-Informed Autoencoder (PIA) for IVIM parameter estimation
"""
import os
import argparse
import torch
from utils import get_batch_ivim, set_seed
from pia import PIA

def train(args):
    # reproducibility
    set_seed(42)

    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate IVIM-specific PIA
    model = PIA(predictor_depth=args.predictor_depth, device=device).to(device)

    # collect trainable parameters: encoder and IVIM predictors
    params = (
        list(model.encoder.parameters()) +
        list(model.D_predictor.parameters()) +
        list(model.D_star_predictor.parameters()) +
        list(model.F_predictor.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=args.lr)

    running_loss = 0.0
    # training loop
    for epoch in range(1, args.num_epochs + 1):
        # generate a batch with optional Gaussian noise during training
        x_noisy, D_true, F_true, x_clean = get_batch_ivim(
            batch_size=args.batch_size,
            noise_sdt=args.noise_sdt
        )
        x_noisy = x_noisy.to(device)
        x_clean = x_clean.to(device)

        optimizer.zero_grad()
        # forward pass: encode and decode
        D_pred, D_star_pred, F_pred = model.encode(x_noisy)
        recon = model.decode(D_pred, D_star_pred, F_pred)

        # reconstruction loss against clean signal
        loss = model.loss_function(recon, x_clean)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # log training progress
        if epoch % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            print(f"Epoch [{epoch}/{args.num_epochs}] - Avg Loss: {avg_loss:.6f}")
            running_loss = 0.0

    # save the trained model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'pia_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model training complete and saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IVIM PIA')
    parser.add_argument('--predictor_depth', type=int, default=2,
                        help='Depth of predictor networks')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for synthetic IVIM data')
    parser.add_argument('--num_epochs', type=int, default=100000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for optimizer')
    parser.add_argument('--noise_sdt', type=float, default=0.04,
                        help='Std dev of Rician noise in training signals')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Print training loss every N epochs')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save trained model')
    args = parser.parse_args()
    train(args)
