#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 22:44:02 2025

@author: leontan
"""

import argparse
import os 


from utils_test import (
    set_seed,
    get_batch_ivim,
    hybrid_ivim_fit,
    ivim_signal,
    get_ivim_scores as get_scores,   
    #get_scores2,                     
    plot_ivim_results as plot_results,
    calculate_ivim_stats as calculate_stats,
    compare_mae, calculate_mae_bias_variance
    # get_batch_2compartment_ivim,  
)

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pia import PIA
import torch.optim as optim
from tqdm import tqdm
from scipy.stats import ttest_ind



parser = argparse.ArgumentParser(description='Monte Carlo Experiments of PIA for diffusion-relaxation model')
parser.add_argument('--sample_size', type=int, default=2500, help='Sample size for in-silica experiments')
parser.add_argument('--noise_sdt', type=float, default=0.02, help='Noise standard deviation')

def main(args):
    set_seed(16)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    print('Starting the In-silica Experiments for Physics-Informed Autoencoder (PIA) for diffusion-relaxation model...')
    
    test_tensor, D_test, F_test, clean = get_batch_ivim(args.sample_size, noise_sdt=args.noise_sdt)
    test = test_tensor.detach().cpu().numpy()

    start = time.time()
    print('Calculating the NLLS fit - this may take a while...(but rejoice, we are solving it with PIA!)')
    D_NLLS, D_star_NLLS, F_NLLS = hybrid_ivim_fit(test)  # Unpack all three diffusion parameters
    end = time.time()
    print(f'NLLS solution takes {end - start:.2f} seconds for {args.sample_size} samples')

    D_NLLS_stacked = np.stack([D_NLLS, D_star_NLLS], axis=1)  # Stack D and D_star for scoring

    get_scores((D_test, F_test), (D_NLLS_stacked, F_NLLS))
    print('Plotting the results...')
    plot_results((D_test, F_test), (D_NLLS_stacked, F_NLLS), method='NLLS')

    print('Now Testing with the PIA model...')
    
    device = torch.device('cpu')  # force CPU since CUDA unavailable, was getting errors with this
    
    model = PIA(predictor_depth=2, device=device).to(device)
    model.load_state_dict(torch.load('pia.pt', map_location=device))
    model.eval()

    test_tensor = test_tensor.to(device)

    start = time.time()
    
    # Unpack three outputs from encode
    D_pred, D_star_pred, F_pred = model.encode(test_tensor)
    end = time.time()
    print(f'PIA takes {end - start:.2f} seconds for {args.sample_size} samples')

    # Stack D_pred and D_star_pred for scoring
    D_pred_stacked = D_pred.detach().cpu().numpy()
    D_star_pred_stacked = D_star_pred.detach().cpu().numpy()
    D_pia_stacked = np.stack([D_pred_stacked, D_star_pred_stacked], axis=1)

    get_scores((D_test, F_test), (D_pia_stacked, F_pred.detach().cpu().numpy()))
    plot_results((D_test, F_test), (D_pia_stacked, F_pred.detach().cpu().numpy()), method='PIA')
    
    
    
    D_test_np = D_test.detach().cpu().numpy() if torch.is_tensor(D_test) else D_test
    F_test_np = F_test.detach().cpu().numpy() if torch.is_tensor(F_test) else F_test
    F_pred_np = F_pred.detach().cpu().numpy() if torch.is_tensor(F_pred) else F_pred

    # Print MAE, Bias, Variance for each param
    def print_raw_metrics(name, y_true, y_pred):
        mae, bias, var = calculate_mae_bias_variance(y_true, y_pred)
        print(f"{name:8s} | MAE: {mae:.4f}, Bias: {bias:.4f}, Var: {var:.4f}")

        print("\n--- Raw Error Metrics ---")
    for i, param in enumerate(["D", "D*", "F"]):
        if param == "F":
            print_raw_metrics("PIA " + param, F_test_np, F_pred_np)
            print_raw_metrics("NLLS " + param, F_test_np, F_NLLS)
        else:
            print_raw_metrics("PIA " + param, D_test_np[:,i], D_pia_stacked[:,i])
            print_raw_metrics("NLLS " + param, D_test_np[:,i], D_NLLS_stacked[:,i])

    # Now run comparison tests
    print('\n--- Comparing the results of PIA and NLLS (p-values) ---')
    calculate_stats(
        (D_test_np, F_test_np),
        (D_pia_stacked, F_pred_np),
        (D_NLLS_stacked, F_NLLS)
        )

    print('Conducting Speed Tests')
    test_tensor, D_test2, F_test2, clean = get_batch_ivim(20000, noise_sdt=args.noise_sdt)
    test = test_tensor.detach().cpu().numpy()

    start = time.time()
    D_hybrid, D_star_hybrid, F_hybrid = hybrid_ivim_fit(test)
    end = time.time()
    print(f'NLLS solution took {end - start:.2f} seconds for 20,000 samples')

    D_hybrid_stacked = np.stack([D_hybrid, D_star_hybrid], axis=1)

    test_tensor = test_tensor.to(device)
    start = time.time()
    
    # Unpack three outputs from encode, then select only D and F for speed print if needed
    D_pia_speed, D_star_pia_speed, F_pia_speed = model.encode(test_tensor)
    end = time.time()
    print(f'PIA took {end - start:.2f} seconds for 20,000 samples')

    print('Done')
    


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)