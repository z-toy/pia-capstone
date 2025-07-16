#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 19:21:24 2025

@author: zoeatoy
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List

class PIA(nn.Module):

    def __init__(self,
                 number_of_signals=8,
                 D_mean=[1.5, 30.0],             # Midpoints of D and D* ranges
                 D_delta=[1.5, 30.0],            # Spread to cover 0–2.2 and 0–60
                 b_values=[0, 5, 50, 100, 200, 500, 800, 1000],
                 hidden_dims: List = None,
                 predictor_depth=1,
                 device='cuda'):
        super().__init__()
        

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.number_of_signals = number_of_signals
        self.register_buffer('D_mean', torch.tensor(D_mean))
        self.register_buffer('D_delta', torch.tensor(D_delta))
        self.b_values = b_values
        self.device = device
        self.relu = nn.ReLU()

        modules = []
        in_channels = number_of_signals
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        # Separate predictors for D and D_star
        D_predictor = []
        for _ in range(predictor_depth):
            D_predictor.append(nn.Linear(hidden_dims[-1], hidden_dims[-1]))
            D_predictor.append(nn.LeakyReLU())
        D_predictor.append(nn.Linear(hidden_dims[-1], 1))  # Output scalar D
        self.D_predictor = nn.Sequential(*D_predictor)

        D_star_predictor = []
        for _ in range(predictor_depth):
            D_star_predictor.append(nn.Linear(hidden_dims[-1], hidden_dims[-1]))
            D_star_predictor.append(nn.LeakyReLU())
        D_star_predictor.append(nn.Linear(hidden_dims[-1], 1))  # Output scalar D*
        self.D_star_predictor = nn.Sequential(*D_star_predictor)

        # F predictor
        F_predictor = []
        for _ in range(predictor_depth):
            F_predictor.append(nn.Linear(hidden_dims[-1], hidden_dims[-1]))
            F_predictor.append(nn.LeakyReLU())
        F_predictor.append(nn.Linear(hidden_dims[-1], 1))
        self.F_predictor = nn.Sequential(*F_predictor)
        
    def encode(self, x):
        """
        Encodes input signals into latent parameters: D, D*, and F separately
        """
        result = self.encoder(x)

        D = self.D_delta[0] * torch.tanh(self.D_predictor(result)).squeeze(dim=1) + self.D_mean[0]  # [batch]
        D_star = self.D_delta[1] * torch.tanh(self.D_star_predictor(result)).squeeze(dim=1) + self.D_mean[1]  # [batch]
        F = self.F_predictor(result).squeeze(dim=1)
        F = torch.sigmoid(F)

        return D, D_star, F

    def decode(self, D, D_star, F):
        """
        Reconstruct signal using IVIM model:
        S(b) = (1 - F) * exp(-b * D) + F * exp(-b * D*)
        """
        signal = torch.zeros((D.shape[0], self.number_of_signals), device=self.device)

        for i, b in enumerate(self.b_values):
            S = (1 - F) * torch.exp(-b * D / 1000) + F * torch.exp(-b * D_star / 1000)
            signal[:, i] = S

        return (1000*signal).to(self.device)
    
    def forward(self, x):
        D, D_star, F = self.encode(x)
        reconstructed = self.decode(D, D_star, F)
        return [reconstructed, x, D, D_star, F]

    def loss_function(self, pred_signal, true_signal, weights=None):
        if weights is not None:
            loss = torch.mean(weights * (pred_signal - true_signal) ** 2)
        else:
            loss = F.mse_loss(pred_signal, true_signal)
        return loss
