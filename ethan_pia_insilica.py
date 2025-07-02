#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo In-Silica Experiments for IVIM PIA vs. NLLS
"""
import argparse
import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ttest_ind, spearmanr

from utils import (
    set_seed,
    get_batch_ivim,
    ivim_fit,
    get_ivim_scores as get_scores,
    plot_ivim_results as plot_results,
    calculate_ivim_stats as calculate_stats
)
from pia import PIA

parser = argparse.ArgumentParser(
    description='Monte Carlo Experiments for IVIM PIA vs. NLLS'
)
parser.add_argument('--sample_size', type=int, default=2500,
                    help='Sample size for in-silica experiments')
parser.add_argument('--noise_std', type=float, default=0.02,
                    help='Noise standard deviation')


def main(args):
    # reproducibility
    set_seed(16)
    os.makedirs('plots', exist_ok=True)

    print('Starting in-silica experiments for IVIM PIA vs. NLLS')

    # Generate test data
    test_tensor, D_test, F_test, clean = get_batch_ivim(
        batch_size=args.sample_size,
        noise_sdt=args.noise_std
    )
    test_np = test_tensor.detach().cpu().numpy()

    # Stage 0: NLLS baseline
    print('Calculating NLLS fit...')
    start = time.time()
    D_nlls, D_star_nlls, F_nlls = ivim_fit(test_np)
    print(f'NLLS took {time.time() - start:.2f}s for {args.sample_size} samples')

    D_nlls_params = np.stack([D_nlls, D_star_nlls], axis=1)
    get_scores((D_test.detach().cpu().numpy(), F_test.detach().cpu().numpy()),
               (D_nlls_params, F_nlls))
    plot_results((D_test.detach().cpu().numpy(), F_test.detach().cpu().numpy()),
                 (D_nlls_params, F_nlls), method='NLLS')

    # Stage 1: PIA inference
    print('Loading trained PIA model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PIA(predictor_depth=2, device=device).to(device)
    model.load_state_dict(torch.load('pia.pt', map_location=device))
    model.eval()

    test_tensor = test_tensor.to(device)
    print('Running PIA inference...')
    start = time.time()
    D_pia, D_star_pia, F_pia = model.encode(test_tensor)
    print(f'PIA took {time.time() - start:.2f}s for {args.sample_size} samples')

    D_pia_np = D_pia.detach().cpu().numpy()
    D_star_pia_np = D_star_pia.detach().cpu().numpy()
    F_pia_np = F_pia.detach().cpu().numpy()
    D_pia_params = np.stack([D_pia_np, D_star_pia_np], axis=1)

    get_scores((D_test.detach().cpu().numpy(), F_test.detach().cpu().numpy()),
               (D_pia_params, F_pia_np))
    plot_results((D_test.detach().cpu().numpy(), F_test.detach().cpu().numpy()),
                 (D_pia_params, F_pia_np), method='PIA')

    # Stage 2: Statistical comparison
    print('Comparing PIA vs NLLS (p-values)')
    calculate_stats(
        (D_test.detach().cpu().numpy(), F_test.detach().cpu().numpy()),
        (D_pia_params, F_pia_np),
        (D_nlls_params, F_nlls)
    )

    # Stage 3: Speed test
    print('Conducting speed test...')
    large_bs = 20000
    large_tensor, _, _, _ = get_batch_ivim(large_bs, noise_sdt=args.noise_std)
    large_np = large_tensor.detach().cpu().numpy()

    start = time.time()
    _ = ivim_fit(large_np)
    print(f'NLLS on {large_bs} took {time.time() - start:.2f}s')

    large_tensor = large_tensor.to(device)
    start = time.time()
    _ = model.encode(large_tensor)
    print(f'PIA on {large_bs} took {time.time() - start:.2f}s')

    # Stage 4: Noise robustness
    print('Stage 4: Noise robustness')
    noise_levels = [2e-4, 5e-4, 7e-4, 1e-3, 2e-3, 5e-3, 7e-3, 1e-2, 2e-2, 5e-2, 7e-2, 1e-1]
    results = {'D': [], 'D*': [], 'F': []}
    B = 100

    for sigma in tqdm(noise_levels):
        # generate clean + Rician noise
        _, D_t_true, F_t_true, clean = get_batch_ivim(B, noise_sdt=0)
        clean_np = clean.detach().cpu().numpy() / 1000
        real = np.random.normal(0, sigma, clean_np.shape)
        imag = np.random.normal(0, sigma, clean_np.shape)
        noisy = np.abs(clean_np + real + 1j * imag)

        # NLLS predictions
        Dh, Dstar_h, Fh = ivim_fit(noisy * 1000)

        # PIA predictions
        inp = 1000 * torch.from_numpy(noisy).float().to(device)
        Dp, Dstar_p, Fp = model.encode(inp)
        Dp_np = Dp.detach().cpu().numpy()
        Dstar_p_np = Dstar_p.detach().cpu().numpy()
        Fp_np = Fp.detach().cpu().numpy()

        # true params
        true_D = D_t_true.detach().cpu().numpy()
        true_F = F_t_true.detach().cpu().numpy()

        # compute metrics per parameter
        for name, y_true, y_pia, y_nlls in [
            ('D',  true_D[:,0],      Dp_np,        Dh),
            ('D*', true_D[:,1],      Dstar_p_np,   Dstar_h),
            ('F',  true_F,           Fp_np,        Fh)
        ]:
            r_pia = spearmanr(y_pia, y_true).statistic
            r_nlls = spearmanr(y_nlls, y_true).statistic
            mae_pia = np.mean(np.abs(y_pia - y_true))
            mae_nlls = np.mean(np.abs(y_nlls - y_true))
            bias_pia = np.mean(y_pia - y_true)
            bias_nlls = np.mean(y_nlls - y_true)
            std_pia = np.std(y_pia - y_true)
            std_nlls = np.std(y_nlls - y_true)

            results[name].append({
                'noise': sigma,
                'pia':   (r_pia,   mae_pia,   bias_pia,   std_pia),
                'nlls':  (r_nlls,  mae_nlls,  bias_nlls,  std_nlls)
            })

    # Plot noise robustness
    for key, data in results.items():
        fig, axs = plt.subplots(2,2,figsize=(12,10))
        metrics = ['corr','mae','bias','std']
        for idx, metric in enumerate(metrics):
            ax = axs[idx//2, idx%2]
            x = [r['noise'] for r in data]
            pia_vals = [r['pia'][idx] for r in data]
            nlls_vals = [r['nlls'][idx] for r in data]

            ax.plot(x, pia_vals, marker='o', label='PIA')
            ax.plot(x, nlls_vals, marker='s', label='NLLS')
            ax.set_xscale('log')
            ax.set_xlabel('Noise Std Dev')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{key} - {metric.capitalize()} vs Noise')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'plots/noise_robustness_{key}.png')
        plt.close(fig)

    print('Noise robustness plots saved in plots/')
    print('Done')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
