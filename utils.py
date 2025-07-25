# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy import stats
import random
import os

def get_batch_ivim(batch_size=16, noise_sdt=0.1, b_values=[0, 5, 50, 100, 200, 500, 800, 1000]):
    if b_values is None:
        b_values = [0, 5, 50, 100, 200, 500, 800, 1000]

    # Generate ground truth parameters
    D = np.random.uniform(0.0, 2.9, batch_size)        # True diffusion
    D_star = np.random.uniform(3.0, 60.0, batch_size)  # Pseudo-diffusion
    F = np.random.uniform(0.0, 1.0, batch_size)        # Perfusion fraction

    signal = np.zeros((batch_size, len(b_values)), dtype=float)

    # Simulate signal using S(b) = (1 - F) * exp(-b * D) + F * exp(-b * D*)
    for sample in range(batch_size):
        for i, b in enumerate(b_values):
            Sb = (1 - F[sample]) * np.exp(-b * D[sample] / 1000) + F[sample] * np.exp(-b * D_star[sample] / 1000)
            signal[sample, i] = Sb

    real = np.random.normal(0, noise_sdt, signal.shape)
    imag = np.random.normal(0, noise_sdt, signal.shape)
    noisy_signal = np.abs(signal + real + 1j * imag)

    # Inputs and targets
    input_tensor = torch.from_numpy(noisy_signal).float()
    clean_signal = torch.from_numpy(signal).float()
    
    # Ground truth parameters (optional, not needed for training)
    D_params = torch.from_numpy(np.stack([D, D_star], axis=1)).float()
    F_tensor = torch.from_numpy(F).float()

    return input_tensor, D_params, F_tensor, clean_signal


def ivim_model(bvals, D, D_star, F):
    """Two-compartment IVIM signal model"""
    return (1 - F) * np.exp(-bvals * D / 1000) + F * np.exp(-bvals * D_star / 1000)

def ivim_fit(signals, bvals=[0, 5, 50, 100, 200, 500, 800, 1000]):
    eps = 1e-7
    num_voxels, acquisitions = signals.shape
    D_vals = np.zeros(num_voxels)
    D_star_vals = np.zeros(num_voxels)
    F_vals = np.zeros(num_voxels)

    for i in range(num_voxels):
        ydata = signals[i]
        try:
            popt, _ = curve_fit(
                ivim_model,
                np.array(bvals),
                ydata,
                p0=[1.0, 30.0, 0.5],                    # Initial guesses: D, D*, F
                bounds=([0.0, 0.0, 0.3], [2.2, 60.0, 0.7]),  # Physiological bounds
                method='trf',
                maxfev=5000
            )
        except RuntimeError:
            popt = [1.0, 30.0, 0.5]  # Default fallback if fitting fails

        D_vals[i], D_star_vals[i], F_vals[i] = popt

    return D_vals, D_star_vals, F_vals

from scipy.stats import spearmanr
import numpy as np
import torch

def get_ivim_scores(true, pred):
    """
    Computes Spearman correlation, MAE, bias, and stddev for IVIM parameters.
    
    Args:
        true: tuple of ground truth tensors/arrays: (D_true [N×2], F_true [N])
        pred: tuple of predicted tensors/arrays:   (D_pred [N×2], F_pred [N])
    """
    D_true, F_true = true
    D_pred, F_pred = pred

    print(f'\t\tCorr\tMAE\tBias\tStddev')

    # Parameter names and values
    param_names = ['D', 'D*', 'F']
    true_params = [D_true[:, 0], D_true[:, 1], F_true]
    pred_params = [D_pred[:, 0], D_pred[:, 1], F_pred]

    for name, x_vals, y_vals in zip(param_names, true_params, pred_params):
        # Convert to numpy if needed
        if isinstance(x_vals, torch.Tensor):
            x_vals = x_vals.detach().cpu().numpy()
        if isinstance(y_vals, torch.Tensor):
            y_vals = y_vals.detach().cpu().numpy()

        corr = round(spearmanr(y_vals, x_vals).statistic, 2)
        mae = np.mean(np.abs(y_vals - x_vals))
        bias = np.mean(y_vals - x_vals)
        std = np.std(y_vals - x_vals)
        
        print(f'{name}\t\t{corr:.2f}\t{mae:.2f}\t{bias:.2f}\t{std:.2f}')

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_ivim_results(true, pred, method='NLLS'):
    """
    Plots 2D KDE scatter plots comparing predicted vs. true values for IVIM parameters:
    D, D*, and F.
    """
    D_true, F_true = true
    D_pred, F_pred = pred

    param_names = ['D', 'D*', 'F']
    ylims = [(0, 2.2), (0, 60), (0.3, 0.7)]
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))

    true_params = [D_true[:, 0], D_true[:, 1], F_true]
    pred_params = [D_pred[:, 0], D_pred[:, 1], F_pred]

    for i in range(3):
        x = true_params[i]
        y = pred_params[i]

        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()

        nbins = 300
        k = gaussian_kde([x, y])
        xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        ax[i].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap="hot", shading='auto')
        ax[i].scatter(x, y, color='white', s=4, alpha=0.5)
        ax[i].plot([x.min(), x.max()], [x.min(), x.max()], 'w--', linewidth=1)

        ax[i].set_title(f'{param_names[i]}', fontsize=20)
        ax[i].set_xlabel('True', fontsize=16)
        ax[i].set_ylabel('Predicted', fontsize=16)
        ax[i].set_xlim(ylims[i])
        ax[i].set_ylim(ylims[i])
        ax[i].xaxis.set_tick_params(labelsize=14)
        ax[i].yaxis.set_tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(f'plots/scatter_{method}.png')
    plt.close(fig)
    
def steigers_z_test(r1, r2, n1, n2):
    """
    Performs Steiger's Z-test for two dependent correlation coefficients sharing one variable in common.

    Args:
    r1 (float): Pearson correlation coefficient for the first comparison.
    r2 (float): Pearson correlation coefficient for the second comparison.
    n1 (int): Sample size for the first comparison.
    n2 (int): Sample size for the second comparison.

    Returns:
    float: Z-score indicating the difference between the two correlation coefficients.
    float: p-value assessing the significance of the Z-score.
    """
    # Fisher Z transformation for each correlation
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)

    # Standard error for each transformed correlation
    se1 = 1 / np.sqrt(n1 - 3)
    se2 = 1 / np.sqrt(n2 - 3)

    # Standard error of the difference
    sed = np.sqrt(se1**2 + se2**2)

    # Z-score
    z = (z1 - z2) / sed

    # Two-tailed p-value
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))

    return z, p

def calculate_mae_bias_variance(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MSE) and Bias.

    Args:
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted or estimated values.

    """
    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean((y_pred - y_true))
    variance = np.std((y_pred - y_true))**2

    return mae, bias, variance

def compare_mae(y_true, y_pred1, y_pred2):

    mae1, bias1, variance1 = calculate_mae_bias_variance(y_true, y_pred1)
    mae2, bias2, variance2 = calculate_mae_bias_variance(y_true, y_pred2)

    # Perform paired t-test
    _, p_value = stats.ttest_rel(np.abs(y_true - y_pred1), np.abs(y_true - y_pred2))
    _, p_value2 = stats.ttest_rel(y_true - y_pred1, y_true - y_pred2)

    # Calculate variances
    var_a = variance1**2
    var_b = variance2**2

    # Calculate F statistic
    F = var_a / var_b
    df1 = len(y_true) - 1  # degrees of freedom for sample 1
    df2 = len(y_true) - 1  # degrees of freedom for sample 2

    # Calculate p-value
    p_value3 = 1 - stats.f.cdf(F, df1, df2) if var_a > var_b else stats.f.cdf(F, df1, df2)

    return p_value, p_value2, p_value3

from scipy.stats import spearmanr

def calculate_ivim_stats(test, pia, NLLS):
    """
    Compare PIA and NLLS predictions vs ground truth for IVIM parameters.
    Prints Steiger's z-test p-value for correlation difference and compare_mae stats.
    
    Args:
        test: tuple (D_true [N,2], F_true [N])
        pia: tuple (D_pia [N,2], F_pia [N])
        NLLS: tuple (D_nlls [N,2], F_nlls [N])
    """
    D_test, F_test = test
    D_pia, F_pia = pia
    D_nlls, F_nlls = NLLS

    param_names = ['D', 'D*', 'F']

    # Stack perfusion fraction with diffusion params for uniform looping
    # D and D* are 2D arrays (N x 2), F is 1D (N,)
    true_params = [D_test[:,0], D_test[:,1], F_test]
    pia_params = [D_pia[:,0], D_pia[:,1], F_pia]
    nlls_params = [D_nlls[:,0], D_nlls[:,1], F_nlls]

    print(f'\tCorr_pval\tMAE_diff\tBias_diff\tStddev_diff')

    for i, name in enumerate(param_names):
        x = true_params[i]
        y_pia = pia_params[i]
        y_nlls = nlls_params[i]

        r_pia = spearmanr(x, y_pia).statistic
        r_nlls = spearmanr(x, y_nlls).statistic
        
        # Assuming compare_mae returns (mae_diff, bias_diff, std_diff)
        p_mae, p_bias, p_var = compare_mae(x, y_pia, y_nlls)

        # Number of samples (assuming same length)
        n = len(x)
        # Compute p-value for difference in correlations via Steiger's Z test
        p_corr = steigers_z_test(r_nlls, r_pia, n, n)[1]

        print(f'{name}\t{p_corr:.5f}\t{p_mae:.5f}\t{p_bias:.5f}\t{p_var:.5f}')

import numpy as np
import torch

def set_seed(seed):
    # TODO add mps seed setting
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    
def ccc(x,y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc