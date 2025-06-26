#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 19:47:50 2025

@author: zoeatoy
"""

from pia import PIA
import torch
import numpy as np
from utils import get_batch_ivim
import argparse


parser = argparse.ArgumentParser(description='Training PIA for diffusion-relaxation model')
parser.add_argument('--predictor_depth', type=int, default=2, help='Depth of the predictor network')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=5000, help='Number of epochs') #500000
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PIA(predictor_depth=args.predictor_depth, device=device).to(device)

    # Collect parameters for optimizer (adjust these attribute names as per your class!)
    params = (
    list(model.encoder.parameters()) +
    list(model.D_predictor.parameters()) +
    list(model.D_star_predictor.parameters()) +
    list(model.F_predictor.parameters())
)

    optimizer = torch.optim.Adam(params, lr=args.lr)

    ctr = 1
    total_loss = 0.0

    for ep in range(args.num_epochs):
        # Unpack 5 returned values from get_batch
        x, _, _, y = get_batch_ivim(args.batch_size)
        
        x, y = x.to(device), y.to(device)  # move to device
        
        optimizer.zero_grad()

        # Encode to get predicted parameters
        D, D_star, F = model.encode(x)

        # Decode to reconstruct signal
        recon = model.decode(D, D_star, F)

        # Compute loss with predicted reconstruction vs true clean signal y
        loss = model.loss_function(recon, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if ep % 10 == 0:
            print(f'Epoch {ep} - Avg Loss: {(total_loss / ctr):.4f}')
        ctr += 1


    torch.save(model.state_dict(), 'pia.pt')
    print('Model training complete and saved to pia.pt')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
