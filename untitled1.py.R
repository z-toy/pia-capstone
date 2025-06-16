import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as stats
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd

# Set style parameters for publication-quality figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Function to load sample data
def load_sample_data():
    """
    This function would normally load your real data.
    For demonstration, we'll create synthetic data that mimics your IVIM dataset.
    """
    # Create a synthetic 200x200 image
    np.random.seed(42)
    
    # Tissue mask with different regions (0=background, 1=adipose, 2=fibroglandular, 3=tumor)
    tissue_mask = np.zeros((200, 200), dtype=int)
    
    # Create background
    tissue_mask[:] = 0
    
    # Create breast shape (circle)
    y, x = np.ogrid[:200, :200]
    center = (100, 100)
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    breast_mask = dist_from_center <= 90
    tissue_mask[breast_mask] = 1  # Adipose tissue
    
    # Create fibroglandular tissue
    fibro_mask = dist_from_center <= 60
    tissue_mask[fibro_mask] = 2
    
    # Create tumor
    tumor_center = (140, 100)
    tumor_dist = np.sqrt((x - tumor_center[0])**2 + (y - tumor_center[1])**2)
    tumor_mask = tumor_dist <= 15
    tissue_mask[tumor_mask] = 3
    
    # Create parameter maps
    # True parameters for each tissue type
    tissue_params = {
        0: {'f': 0, 'D': 0, 'Dstar': 0},  # Background
        1: {'f': 0.035, 'D': 2.1e-3, 'Dstar': 12.0e-3},  # Adipose
        2: {'f': 0.105, 'D': 1.5e-3, 'Dstar': 18.0e-3},  # Fibroglandular
        3: {'f': 0.060, 'D': 0.9e-3, 'Dstar': 10.0e-3}   # Tumor
    }
    
    # Create parameter maps
    f_map = np.zeros((200, 200))
    D_map = np.zeros((200, 200))
    Dstar_map = np.zeros((200, 200))
    
    for tissue_type, params in tissue_params.items():
        mask = tissue_mask == tissue_type
        f_map[mask] = params['f']
        D_map[mask] = params['D']
        Dstar_map[mask] = params['Dstar']
    
    # Add some random variation (heterogeneity)
    f_noise = np.random.normal(0, 0.005, (200, 200))
    D_noise = np.random.normal(0, 0.0001, (200, 200))
    Dstar_noise = np.random.normal(0, 0.001, (200, 200))
    
    f_map += f_noise
    D_map += D_noise
    Dstar_map += Dstar_noise
    
    # Ensure parameters stay in valid ranges
    f_map = np.clip(f_map, 0, 1)
    D_map = np.clip(D_map, 0, 3e-3)
    Dstar_map = np.clip(Dstar_map, 3e-3, 50e-3)
    
    # Set background to zero
    background = ~breast_mask
    f_map[background] = 0
    D_map[background] = 0
    Dstar_map[background] = 0
    
    # Generate estimated parameter maps for different methods
    methods = ["NLLS", "Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6"]
    error_levels = {
        "NLLS": {'f': 0.30, 'D': 0.22, 'Dstar': 0.49},
        "Model 1": {'f': 0.19, 'D': 0.15, 'Dstar': 0.32},
        "Model 2": {'f': 0.17, 'D': 0.14, 'Dstar': 0.29},
        "Model 3": {'f': 0.23, 'D': 0.17, 'Dstar': 0.37},
        "Model 4": {'f': 0.20, 'D': 0.16, 'Dstar': 0.34},
        "Model 5": {'f': 0.18, 'D': 0.15, 'Dstar': 0.31},
        "Model 6": {'f': 0.15, 'D': 0.12, 'Dstar': 0.27}
    }
    
    estimated_maps = {}
    for method in methods:
        f_err = np.random.normal(0, error_levels[method]['f'] * f_map, (200, 200))
        D_err = np.random.normal(0, error_levels[method]['D'] * D_map, (200, 200))
        Dstar_err = np.random.normal(0, error_levels[method]['Dstar'] * Dstar_map, (200, 200))
        
        # More structured error for NLLS (speckle noise pattern)
        if method == "NLLS":
            speckle = np.random.uniform(0, 1, (200, 200)) < 0.2
            f_err[speckle] *= 3
            D_err[speckle] *= 3
            Dstar_err[speckle] *= 3
        
        # Smoother errors for deep learning models
        if method in ["Model 1", "Model 2", "Model 4", "Model 5", "Model 6"]:
            from scipy.ndimage import gaussian_filter
            f_err = gaussian_filter(f_err, sigma=1.0)
            D_err = gaussian_filter(D_err, sigma=1.0)
            Dstar_err = gaussian_filter(Dstar_err, sigma=1.0)
        
        # Model 6 has more accurate tumor boundary
        if method == "Model 6":
            tumor_boundary = np.logical_and(tumor_dist >= 14, tumor_dist <= 16)
            f_err[tumor_boundary] *= 0.5
            D_err[tumor_boundary] *= 0.5
            Dstar_err[tumor_boundary] *= 0.5
        
        estimated_maps[method] = {
            'f': np.clip(f_map + f_err, 0, 1),
            'D': np.clip(D_map + D_err, 0, 3e-3),
            'Dstar': np.clip(Dstar_map + Dstar_err, 3e-3, 50e-3)
        }
    
    # Generate DWI signal data
    b_values = np.array([0, 5, 50, 100, 200, 500, 800, 1000])
    
    # Create DWI signals for each voxel
    dwi_signals = np.zeros((200, 200, len(b_values)))
    
    for i in range(200):
        for j in range(200):
            if breast_mask[i, j]:
                f = f_map[i, j]
                D = D_map[i, j]
                Dstar = Dstar_map[i, j]
                
                # IVIM signal equation: S(b) = S0 * ((1-f) * exp(-b*D) + f * exp(-b*Dstar))
                for k, b in enumerate(b_values):
                    dwi_signals[i, j, k] = (1-f) * np.exp(-b * D) + f * np.exp(-b * Dstar)
    
    # Add noise to DWI signals
    noise_level = 0.02
    for k in range(len(b_values)):
        noise = np.random.normal(0, noise_level, (200, 200))
        dwi_signals[:, :, k] = np.clip(dwi_signals[:, :, k] + noise, 0, None)
    
    # Generate VAE uncertainty maps (higher uncertainty for D*)
    uncertainty_maps = {
        'f': np.zeros((200, 200)),
        'D': np.zeros((200, 200)),
        'Dstar': np.zeros((200, 200))
    }
    
    # Higher uncertainty at tumor boundaries and in low SNR regions
    uncertainty_maps['f'] = 0.02 + 0.03 * np.exp(-((tumor_dist - 15) / 5)**2)
    uncertainty_maps['D'] = 0.0002 + 0.0003 * np.exp(-((tumor_dist - 15) / 5)**2)
    uncertainty_maps['Dstar'] = 0.002 + 0.003 * np.exp(-((tumor_dist - 15) / 5)**2)
    
    # Higher uncertainty for farther regions due to lower SNR
    radial_falloff = np.clip(dist_from_center / 90, 0, 1) * 0.5
    uncertainty_maps['f'] += radial_falloff * 0.02
    uncertainty_maps['D'] += radial_falloff * 0.0002
    uncertainty_maps['Dstar'] += radial_falloff * 0.002
    
    # Set background uncertainty to zero
    for param in uncertainty_maps:
        uncertainty_maps[param][background] = 0
    
    # Create error maps (absolute difference between true and estimated)
    error_maps = {}
    for method in methods:
        error_maps[method] = {
            'f': np.abs(estimated_maps[method]['f'] - f_map),
            'D': np.abs(estimated_maps[method]['D'] - D_map),
            'Dstar': np.abs(estimated_maps[method]['Dstar'] - Dstar_map)
        }
    
    # Model performance data across b-values
    b_value_performance = {}
    for method in methods:
        b_value_performance[method] = []
        for i, b in enumerate(b_values):
            # Higher error at higher b-values for NLLS, but more consistent for deep learning
            if method == "NLLS":
                error = 0.15 + 0.05 * np.sqrt(b/1000)
            else:
                # Get method index for scaling
                method_idx = methods.index(method)
                base_error = 0.15 - 0.01 * method_idx
                error = base_error + 0.02 * np.sqrt(b/1000)
            
            b_value_performance[method].append(error)
    
    # Generate calibration data for uncertainty quantification
    calibration_data = {}
    std_levels = np.linspace(0.01, 0.1, 10)
    for param in ['f', 'D', 'Dstar']:
        # For each std level, compute how often the true value falls within the predicted CI
        calibration_data[param] = []
        for std in std_levels:
            # Perfect calibration would be a diagonal line
            # Add some deviation from perfect
            if param == 'f':
                coverage = std * 2 * 0.95 + np.random.normal(0, 0.02)
            elif param == 'D':
                coverage = std * 2 * 0.98 + np.random.normal(0, 0.015)
            else:  # Dstar
                coverage = std * 2 * 0.9 + np.random.normal(0, 0.03)
            
            calibration_data[param].append(np.clip(coverage, 0, 1))
            
    return {
        'tissue_mask': tissue_mask,
        'breast_mask': breast_mask,
        'tumor_mask': tumor_mask,
        'ground_truth': {'f': f_map, 'D': D_map, 'Dstar': Dstar_map},
        'estimated_maps': estimated_maps,
        'dwi_signals': dwi_signals,
        'b_values': b_values,
        'uncertainty_maps': uncertainty_maps,
        'error_maps': error_maps,
        'b_value_performance': b_value_performance,
        'calibration_data': calibration_data,
        'std_levels': std_levels
    }

# 1. Parameter Maps Comparison Visualization
def visualize_parameter_maps(data, save_path=None):
    """
    Create a visualization comparing parameter maps across different methods.
    """
    # Define methods and parameters to display
    methods = ['ground_truth', 'NLLS', 'Model 1', 'Model 6']
    params = ['f', 'D', 'Dstar']
    param_names = ['Perfusion Fraction (f)', 'Diffusion Coefficient (D)', 'Pseudo-Diffusion (D*)']
    
    # Create figure
    fig, axes = plt.subplots(len(params), len(methods), figsize=(16, 12))
    
    # Custom colormaps for each parameter
    cmaps = {
        'f': plt.cm.plasma,
        'D': plt.cm.viridis,
        'Dstar': plt.cm.magma
    }
    
    # Value ranges for each parameter
    vranges = {
        'f': (0, 0.15),
        'D': (0, 2.5e-3),
        'Dstar': (0, 25e-3)
    }
    
    for i, param in enumerate(params):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            
            # Get the appropriate map
            if method == 'ground_truth':
                parameter_map = data['ground_truth'][param]
                title = 'Ground Truth'
            else:
                parameter_map = data['estimated_maps'][method][param]
                title = method
            
            # Create mask for non-zero values
            mask = data['breast_mask']
            
            # Apply mask
            masked_map = np.ma.masked_where(~mask, parameter_map)
            
            # Plot the parameter map
            im = ax.imshow(
                masked_map, 
                cmap=cmaps[param], 
                vmin=vranges[param][0], 
                vmax=vranges[param][1],
                interpolation='none'
            )
            
            # Add tumor boundary outline
            from scipy.ndimage import binary_dilation
            tumor_edge = binary_dilation(data['tumor_mask']) ^ data['tumor_mask']
            ax.contour(tumor_edge, levels=[0.5], colors='white', linewidths=1.5, alpha=0.7)
            
            # Set title and remove axis ticks
            if i == 0:
                ax.set_title(title, fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(param_names[i], fontsize=14)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 2. Signal Decay Curves Visualization
def visualize_signal_decay(data, save_path=None):
    """
    Create a visualization showing signal decay curves for different tissue types.
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Get b-values
    b_values = data['b_values']
    
    # Tissue types and colors
    tissue_types = {
        'Adipose': {'color': 'lightsalmon', 'mask': data['tissue_mask'] == 1},
        'Fibroglandular': {'color': 'lightgreen', 'mask': data['tissue_mask'] == 2},
        'Tumor': {'color': 'crimson', 'mask': data['tissue_mask'] == 3}
    }
    
    # Plot linear scale
    ax = axes[0]
    for tissue_name, tissue_info in tissue_types.items():
        # Get average signal for this tissue type
        mask = tissue_info['mask']
        signal = np.mean(data['dwi_signals'][mask], axis=0)
        
        # Plot the signal decay
        ax.plot(b_values, signal, 'o-', linewidth=2, label=tissue_name, color=tissue_info['color'])
        
        # Add shaded area for signal variability
        signal_std = np.std(data['dwi_signals'][mask], axis=0)
        ax.fill_between(
            b_values, 
            signal - signal_std, 
            signal + signal_std, 
            alpha=0.2, 
            color=tissue_info['color']
        )
    
    ax.set_xlabel('b-value (s/mm²)', fontsize=14)
    ax.set_ylabel('Normalized Signal Intensity', fontsize=14)
    ax.set_title('IVIM Signal Decay Curves', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    # Plot semi-log scale
    ax = axes[1]
    for tissue_name, tissue_info in tissue_types.items():
        # Get average signal for this tissue type
        mask = tissue_info['mask']
        signal = np.mean(data['dwi_signals'][mask], axis=0)
        
        # Plot the signal decay
        ax.semilogy(b_values, signal, 'o-', linewidth=2, label=tissue_name, color=tissue_info['color'])
        
        # Add shaded area for signal variability
        signal_std = np.std(data['dwi_signals'][mask], axis=0)
        ax.fill_between(
            b_values, 
            signal - signal_std, 
            signal + signal_std, 
            alpha=0.2, 
            color=tissue_info['color']
        )
    
    ax.set_xlabel('b-value (s/mm²)', fontsize=14)
    ax.set_ylabel('Normalized Signal Intensity (log scale)', fontsize=14)
    ax.set_title('IVIM Signal Decay Curves (Semi-log)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    # Add annotations explaining bi-exponential behavior
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, edgecolor='gray')
    axes[1].annotate(
        "Perfusion-dominated\nrapid decay at low b-values",
        xy=(60, 0.7), xytext=(200, 0.8),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        bbox=bbox_props, fontsize=12, ha='center'
    )
    
    axes[1].annotate(
        "Diffusion-dominated\ngradual decay at high b-values",
        xy=(800, 0.4), xytext=(600, 0.3),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        bbox=bbox_props, fontsize=12, ha='center'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 3. Model Performance Comparison
def visualize_model_performance(data, save_path=None):
    """
    Create a visualization comparing performance of different models.
    """
    # Extract relevant data
    methods = ["NLLS", "Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6"]
    params = ['f', 'D', 'Dstar']
    param_names = ['Perfusion Fraction (f)', 'Diffusion Coefficient (D)', 'Pseudo-Diffusion (D*)']
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    
    # Define colors for methods
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    
    # 1. Parameter-specific rRMSE bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Compute rRMSE for each parameter and method
    rrmse_data = {param: [] for param in params}
    
    for method in methods:
        for param in params:
            # Get true and estimated maps
            true_map = data['ground_truth'][param]
            est_map = data['estimated_maps'][method][param]
            
            # Compute rRMSE within breast mask
            mask = data['breast_mask']
            true_values = true_map[mask]
            est_values = est_map[mask]
            
            # Compute relative RMSE
            mse = np.mean((true_values - est_values) ** 2)
            rmse = np.sqrt(mse)
            rrmse = rmse / np.sqrt(np.mean(true_values ** 2))
            
            rrmse_data[param].append(rrmse)
    
    # Plot grouped bar chart
    bar_width = 0.25
    x = np.arange(len(methods))
    
    for i, param in enumerate(params):
        offset = (i - 1) * bar_width
        ax1.bar(x + offset, rrmse_data[param], width=bar_width, label=param_names[i], 
               color=plt.cm.tab10(i), alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Method', fontsize=14)
    ax1.set_ylabel('Relative RMSE', fontsize=14)
    ax1.set_title('Parameter-Specific rRMSE by Method', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Tissue-specific improvement heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Define tissue types and methods to display
    tissue_types = ['Adipose', 'Fibroglandular', 'Tumor']
    display_methods = ["Model 1", "Model 2", "Model 4", "Model 6"]
    
    # Create a matrix of improvement percentages compared to NLLS
    improvement_matrix = np.zeros((len(tissue_types), len(display_methods)))
    
    # Tissue masks
    tissue_masks = [
        data['tissue_mask'] == 1,  # Adipose
        data['tissue_mask'] == 2,  # Fibroglandular
        data['tissue_mask'] == 3   # Tumor
    ]
    
    # Compute average improvement for each tissue and method
    for i, tissue_mask in enumerate(tissue_masks):
        for j, method in enumerate(display_methods):
            # Compute average improvement across all parameters
            improvement = 0
            for param in params:
                nlls_error = np.mean(data['error_maps']['NLLS'][param][tissue_mask])
                method_error = np.mean(data['error_maps'][method][param][tissue_mask])
                param_improvement = (nlls_error - method_error) / nlls_error * 100
                improvement += param_improvement
            
            improvement_matrix[i, j] = improvement / len(params)
    
    # Plot heatmap
    im = ax2.imshow(improvement_matrix, cmap='YlGn', aspect='auto', vmin=20, vmax=60)
    
    # Add text annotations
    for i in range(len(tissue_types)):
        for j in range(len(display_methods)):
            ax2.text(j, i, f"{improvement_matrix[i, j]:.1f}%", 
                    ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, label='Improvement over NLLS (%)')
    
    # Set labels
    ax2.set_xticks(np.arange(len(display_methods)))
    ax2.set_yticks(np.arange(len(tissue_types)))
    ax2.set_xticklabels(display_methods)
    ax2.set_yticklabels(tissue_types)
    
    ax2.set_title('Improvement by Tissue Type', fontsize=16, fontweight='bold')
    
    # 3. Overall performance radar chart
    ax3 = fig.add_subplot(gs[1, 0], polar=True)
    
    # Define metrics for radar chart
    metrics = ['Overall\nAccuracy', 'Tumor\nAccuracy', 'Computational\nEfficiency', 
              'Generalization', 'Boundary\nPreservation', 'Noise\nRobustness']
    num_metrics = len(metrics)
    
    # Synthetic performance scores (higher is better)
    scores = {
        'NLLS':    [0.40, 0.45, 0.70, 0.60, 0.30, 0.25],
        'Model 1': [0.70, 0.75, 0.90, 0.65, 0.75, 0.70],
        'Model 2': [0.75, 0.80, 0.90, 0.80, 0.75, 0.75],
        'Model 3': [0.65, 0.70, 0.90, 0.75, 0.70, 0.65],
        'Model 4': [0.70, 0.75, 0.85, 0.80, 0.80, 0.75],
        'Model 5': [0.75, 0.80, 0.85, 0.75, 0.75, 0.80],
        'Model 6': [0.85, 0.90, 0.80, 0.85, 0.85, 0.85]
    }
    
    # Plot radar chart
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Plot each method
    for i, method in enumerate(methods):
        method_scores = scores[method]
        method_scores += method_scores[:1]  # Close the polygon
        
        ax3.plot(angles, method_scores, 'o-', linewidth=2, label=method, color=colors[i], markersize=6)
        ax3.fill(angles, method_scores, color=colors[i], alpha=0.1)
    
    # Set labels
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(metrics, fontsize=12)
    
    # Set radial limits
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # Add legend
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    ax3.set_title('Multi-dimensional Performance Comparison', fontsize=16, fontweight='bold')
    
    # 4. Error distribution violin plot
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Compile error data
    error_data = []
    error_labels = []
    
    # Focus on tumor region for D parameter
    tumor_mask = data['tumor_mask']
    param = 'D'  # Focus on diffusion coefficient
    
    # Selected methods to display
    display_methods = ["NLLS", "Model 1", "Model 6"]
    
    for method in display_methods:
        errors = data['error_maps'][method][param][tumor_mask]
        error_data.append(errors)
        error_labels.append(method)
    
    # Plot violin plots
    vp = ax4.violinplot(error_data, showmedians=True, showextrema=False)
    
    # Customize violin plots
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[methods.index(display_methods[i])])
        body.set_alpha(0.7)
        body.set_edgecolor('black')
        body.set_linewidth(1)
    
    # Add box plots inside violins
    bp = ax4.boxplot(error_data, positions=np.arange(1, len(display_methods) + 1), 
                   widths=0.15, patch_artist=True, showfliers=False)
    
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor='white', alpha=0.7)
    
    # Set labels
    ax4.set_xticks(np.arange(1, len(display_methods) + 1))
    ax4.set_xticklabels(display_methods)
    ax4.set_ylabel('Absolute Error in D (mm²/s)', fontsize=14)
    ax4.set_title(f'Error Distribution in Tumor Region', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 4. Visual comparison of parameter error maps
def visualize_error_maps(data, save_path=None):
    """
    Create a visualization of parameter estimation error maps.
    """
    # Define methods and parameters to display
    methods = ['NLLS', 'Model 1', 'Model 6']
    params = ['f', 'D', 'Dstar']
    param_names = ['Perfusion Fraction (f)', 'Diffusion Coefficient (D)', 'Pseudo-Diffusion (D*)']
    
    # Create figure
    fig, axes = plt.subplots(len(params), len(methods), figsize=(16, 12))
    
    # Error scale factors for visualization
    error_scale = {
        'f': 5,  # Multiply by 5 to make errors more visible
        'D': 1000,  # Convert to 10^-3 mm²/s
        'Dstar': 200  # Scale to similar range
    }
    
    # Value ranges for each parameter's error
    vranges = {
        'f': (0, 0.10),
        'D': (0, 0.5e-3),
        'Dstar': (0, 2.5e-3)
    }
    
    for i, param in enumerate(params):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            
            # Get the error map
            error_map = data['error_maps'][method][param]
            
            # Create mask for non-zero values
            mask = data['breast_mask']
            
            # Apply mask
            masked_map = np.ma.masked_where(~mask, error_map)
            
            # Plot the error map
            im = ax.imshow(
                masked_map, 
                cmap='hot', 
                vmin=vranges[param][0], 
                vmax=vranges[param][1],
                interpolation='none'
            )
            
            # Add tumor boundary outline
            from scipy.ndimage import binary_dilation
            tumor_edge = binary_dilation(data['tumor_mask']) ^ data['tumor_mask']
            ax.contour(tumor_edge, levels=[0.5], colors='white', linewidths=1.5, alpha=0.7)
            
            # Set title and remove axis ticks
            if i == 0:
                ax.set_title(method, fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(param_names[i], fontsize=14)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            if param == 'f':
                cbar.set_label('Error')
            elif param == 'D':
                cbar.set_label('Error (×10⁻³ mm²/s)')
            else:
                cbar.set_label('Error (×10⁻³ mm²/s)')
    
    # Add a title to the figure
    fig.suptitle('Parameter Estimation Error Maps', fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 5. Uncertainty maps from variational model
def visualize_uncertainty_maps(data, save_path=None):
    """
    Create a visualization of uncertainty maps from the variational model.
    """
    # Define parameters to display
    params = ['f', 'D', 'Dstar']
    param_names = ['Perfusion Fraction (f)', 'Diffusion Coefficient (D)', 'Pseudo-Diffusion (D*)']
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Custom colormaps and value ranges for each parameter
    cmaps = {
        'f': plt.cm.plasma,
        'D': plt.cm.viridis,
        'Dstar': plt.cm.magma
    }
    
    # Value ranges for parameters and uncertainties
    vranges = {
        'f': (0, 0.15),
        'D': (0, 2.5e-3),
        'Dstar': (0, 25e-3)
    }
    
    vranges_uncertainty = {
        'f': (0, 0.05),
        'D': (0, 0.5e-3),
        'Dstar': (0, 5e-3)
    }
    
    # First row: parameter estimates
    for i, param in enumerate(params):
        ax = axes[0, i]
        
        # Get parameter map from Model 5 (VAE)
        parameter_map = data['estimated_maps']['Model 5'][param]
        
        # Create mask for non-zero values
        mask = data['breast_mask']
        
        # Apply mask
        masked_map = np.ma.masked_where(~mask, parameter_map)
        
        # Plot the parameter map
        im = ax.imshow(
            masked_map, 
            cmap=cmaps[param], 
            vmin=vranges[param][0], 
            vmax=vranges[param][1],
            interpolation='none'
        )
        
        # Add tumor boundary outline
        from scipy.ndimage import binary_dilation
        tumor_edge = binary_dilation(data['tumor_mask']) ^ data['tumor_mask']
        ax.contour(tumor_edge, levels=[0.5], colors='white', linewidths=1.5, alpha=0.7)
        
        # Set title and remove axis ticks
        ax.set_title(f'Model 5: {param_names[i]}', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    # Second row: uncertainty maps
    for i, param in enumerate(params):
        ax = axes[1, i]
        
        # Get uncertainty map
        uncertainty_map = data['uncertainty_maps'][param]
        
        # Create mask for non-zero values
        mask = data['breast_mask']
        
        # Apply mask
        masked_map = np.ma.masked_where(~mask, uncertainty_map)
        
        # Plot the uncertainty map
        im = ax.imshow(
            masked_map, 
            cmap='Reds', 
            vmin=vranges_uncertainty[param][0], 
            vmax=vranges_uncertainty[param][1],
            interpolation='none'
        )
        
        # Add tumor boundary outline
        ax.contour(tumor_edge, levels=[0.5], colors='white', linewidths=1.5, alpha=0.7)
        
        # Set title and remove axis ticks
        ax.set_title(f'Uncertainty (Standard Deviation) in {param_names[i]}', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    # Add annotations highlighting regions of high uncertainty
    # Find tumor boundary region
    kernel = np.ones((5, 5), np.uint8)
    tumor_boundary = binary_dilation(data['tumor_mask'], kernel) ^ data['tumor_mask']
    boundary_y, boundary_x = np.where(tumor_boundary)
    
    if len(boundary_y) > 0:
        idx = np.random.randint(0, len(boundary_y))
        boundary_point = (boundary_x[idx], boundary_y[idx])
        
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, edgecolor='gray')
        
        axes[1, 0].annotate(
            "Higher uncertainty\nat tumor boundary",
            xy=boundary_point, xytext=(boundary_point[0]-50, boundary_point[1]-30),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            bbox=bbox_props, fontsize=12, ha='center'
        )
    
    # Find peripheral region with higher uncertainty
    y, x = np.ogrid[:200, :200]
    center = (100, 100)
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    peripheral_mask = (dist_from_center > 70) & (dist_from_center < 85) & data['breast_mask']
    periph_y, periph_x = np.where(peripheral_mask)
    
    if len(periph_y) > 0:
        idx = np.random.randint(0, len(periph_y))
        periph_point = (periph_x[idx], periph_y[idx])
        
        axes[1, 2].annotate(
            "Higher uncertainty in\nperipheral regions (lower SNR)",
            xy=periph_point, xytext=(periph_point[0]-20, periph_point[1]+40),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            bbox=bbox_props, fontsize=12, ha='center'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 6. Mixture-of-Experts Selection Map
def visualize_moe_selection(data, save_path=None):
    """
    Create a visualization of the Mixture-of-Experts model selection.
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Get breast mask
    mask = data['breast_mask']
    
    # Create a synthetic expert selection map
    # 1: Model 1 (Supervised), 2: Model 2 (Self-Supervised), 3: Model 5 (VAE)
    expert_map = np.zeros((200, 200), dtype=int)
    
    # Random selection based on tissue type but with patterns
    np.random.seed(42)
    
    # Model 1 tends to be selected more in normal tissue
    fibro_mask = data['tissue_mask'] == 2
    expert_map[fibro_mask] = np.random.choice([1, 2, 3], size=np.sum(fibro_mask), p=[0.5, 0.3, 0.2])
    
    # Model 2 tends to be better for adipose tissue
    adipose_mask = data['tissue_mask'] == 1
    expert_map[adipose_mask] = np.random.choice([1, 2, 3], size=np.sum(adipose_mask), p=[0.3, 0.6, 0.1])
    
    # Model 5 (VAE) tends to be selected more often in tumor and uncertain areas
    tumor_mask = data['tissue_mask'] == 3
    expert_map[tumor_mask] = np.random.choice([1, 2, 3], size=np.sum(tumor_mask), p=[0.2, 0.3, 0.5])
    
    # Add some spatial correlation (smoothing)
    from scipy.ndimage import median_filter
    expert_map = median_filter(expert_map, size=3)
    
    # Ensure outside of breast is zero
    expert_map[~mask] = 0
    
    # Create a custom colormap for the expert map
    colors = ['white', 'forestgreen', 'royalblue', 'crimson']
    expert_cmap = ListedColormap(colors)
    
    # Plot the expert selection map
    ax = axes[0, 0]
    im = ax.imshow(expert_map, cmap=expert_cmap, vmin=0, vmax=3, interpolation='none')
    
    # Add tumor boundary outline
    from scipy.ndimage import binary_dilation
    tumor_edge = binary_dilation(data['tumor_mask']) ^ data['tumor_mask']
    ax.contour(tumor_edge, levels=[0.5], colors='black', linewidths=1.5, alpha=0.7)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=[0.5, 1.5, 2.5])
    cbar.set_ticklabels(['Background', 'Model 1', 'Model 2', 'Model 5'])
    
    ax.set_title('Expert Selection Map', fontsize=14, fontweight='bold')
    
    # Create a residual error map (error for the selected expert at each voxel)
    residual_map = np.zeros((200, 200))
    
    # Use D parameter for residual
    param = 'D'
    
    # Compute residual for the selected expert
    for i in range(200):
        for j in range(200):
            if mask[i, j]:
                expert = expert_map[i, j]
                if expert == 1:
                    method = 'Model 1'
                elif expert == 2:
                    method = 'Model 2'
                elif expert == 3:
                    method = 'Model 5'
                else:
                    continue
                
                residual_map[i, j] = data['error_maps'][method][param][i, j]
    
    # Plot the residual map
    ax = axes[0, 1]
    masked_residual = np.ma.masked_where(~mask, residual_map)
    im = ax.imshow(masked_residual, cmap='hot', vmin=0, vmax=0.5e-3, interpolation='none')
    
    # Add tumor boundary outline
    ax.contour(tumor_edge, levels=[0.5], colors='white', linewidths=1.5, alpha=0.7)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Residual Error (mm²/s)')
    
    ax.set_title('Residual Error Map', fontsize=14, fontweight='bold')
    
    # Plot expert selection distributions by tissue type
    ax = axes[1, 0]
    
    # Count selection frequencies by tissue type
    tissue_types = ['Adipose', 'Fibroglandular', 'Tumor']
    experts = ['Model 1', 'Model 2', 'Model 5']
    
    # Tissue masks
    tissue_masks = [
        data['tissue_mask'] == 1,  # Adipose
        data['tissue_mask'] == 2,  # Fibroglandular
        data['tissue_mask'] == 3   # Tumor
    ]
    
    # Count selections
    selection_counts = np.zeros((len(tissue_types), len(experts)))
    
    for i, t_mask in enumerate(tissue_masks):
        for e in range(1, 4):  # Expert values 1, 2, 3
            selection_counts[i, e-1] = np.sum((expert_map == e) & t_mask)
        
        # Normalize to percentages
        if np.sum(selection_counts[i, :]) > 0:
            selection_counts[i, :] = selection_counts[i, :] / np.sum(selection_counts[i, :]) * 100
    
    # Create grouped bar chart
    x = np.arange(len(tissue_types))
    width = 0.25
    
    for i, expert in enumerate(experts):
        ax.bar(x + (i - 1) * width, selection_counts[:, i], width, label=expert, 
              color=colors[i+1], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add labels and legend
    ax.set_xlabel('Tissue Type', fontsize=14)
    ax.set_ylabel('Selection Percentage (%)', fontsize=14)
    ax.set_title('Expert Selection by Tissue Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tissue_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotations explaining rationale for selections
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, edgecolor='gray')
    
    # Plot performance improvement from MoE
    ax = axes[1, 1]
    
    # Compare MoE with individual models for each parameter
    params = ['f', 'D', 'Dstar']
    methods = ['Model 1', 'Model 2', 'Model 5', 'Model 6']
    
    # Compute average error reduction vs NLLS
    error_reduction = np.zeros((len(params), len(methods)))
    
    for i, param in enumerate(params):
        for j, method in enumerate(methods):
            # Get average error
            nlls_error = np.mean(data['error_maps']['NLLS'][param][mask])
            method_error = np.mean(data['error_maps'][method][param][mask])
            
            # Compute error reduction percentage
            error_reduction[i, j] = (nlls_error - method_error) / nlls_error * 100
    
    # Create grouped bar chart
    x = np.arange(len(params))
    width = 0.2
    
    for i, method in enumerate(methods):
        ax.bar(x + (i - 1.5) * width, error_reduction[:, i], width, label=method, 
              alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add labels and legend
    ax.set_xlabel('Parameter', fontsize=14)
    ax.set_ylabel('Error Reduction vs NLLS (%)', fontsize=14)
    ax.set_title('MoE Performance Improvement', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add arrow pointing to MoE improvement
    ax.annotate(
        "MoE outperforms\nindividual experts",
        xy=(1, error_reduction[1, 3]), xytext=(1.5, error_reduction[1, 3] - 10),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        bbox=bbox_props, fontsize=12
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 7. Performance across b-values
def visualize_b_value_performance(data, save_path=None):
    """
    Create a visualization showing model performance across different b-values.
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Get b-values
    b_values = data['b_values']
    
    # Plot performance across b-values
    ax = axes[0]
    
    # Selected methods to display
    methods = ["NLLS", "Model 1", "Model 2", "Model 6"]
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(methods):
        performance = data['b_value_performance'][method]
        ax.plot(b_values, performance, 'o-', linewidth=2, label=method, color=colors[i], markersize=6)
    
    ax.set_xlabel('b-value (s/mm²)', fontsize=14)
    ax.set_ylabel('Relative Error', fontsize=14)
    ax.set_title('Model Performance vs. b-value', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=12)
    
    # Add annotations
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, edgecolor='gray')
    
    ax.annotate(
        "Deep learning models maintain\naccuracy at high b-values",
        xy=(800, data['b_value_performance']["Model 6"][-1]), 
        xytext=(600, data['b_value_performance']["Model 6"][-1] - 0.05),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        bbox=bbox_props, fontsize=12
    )
    
    ax.annotate(
        "Conventional NLLS performance\ndegrades at high b-values",
        xy=(800, data['b_value_performance']["NLLS"][-1]), 
        xytext=(600, data['b_value_performance']["NLLS"][-1] + 0.05),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        bbox=bbox_props, fontsize=12
    )
    
    # Plot SNR vs b-value
    ax = axes[1]
    
    # Compute average SNR for each b-value
    snr_values = []
    for i, b in enumerate(b_values):
        signal = np.mean(data['dwi_signals'][data['breast_mask'], i])
        noise_level = 0.02  # Constant noise level
        snr = signal / noise_level
        snr_values.append(snr)
    
    # Plot SNR curve
    ax.semilogy(b_values, snr_values, 'o-', linewidth=3, color='crimson', markersize=8)
    
    ax.set_xlabel('b-value (s/mm²)', fontsize=14)
    ax.set_ylabel('Signal-to-Noise Ratio (log scale)', fontsize=14)
    ax.set_title('SNR vs. b-value', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add critical SNR threshold
    ax.axhline(y=5, linestyle='--', color='gray', alpha=0.7, linewidth=2)
    ax.text(50, 5.5, 'Critical SNR Threshold', fontsize=12, color='gray')
    
    # Add annotations
    ax.annotate(
        "Low SNR at high b-values\nchallenges parameter estimation",
        xy=(800, snr_values[-1]), 
        xytext=(500, snr_values[-1] * 2),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        bbox=bbox_props, fontsize=12
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 8. Uncertainty calibration plot
def visualize_uncertainty_calibration(data, save_path=None):
    """
    Create a visualization of uncertainty calibration for the VAE model.
    """
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Parameters to display
    params = ['f', 'D', 'Dstar']
    param_names = ['Perfusion Fraction (f)', 'Diffusion Coefficient (D)', 'Pseudo-Diffusion (D*)']
    param_colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    # Get calibration data
    std_levels = data['std_levels']
    
    for i, param in enumerate(params):
        ax = axes[i]
        
        # Get calibration data for this parameter
        actual_coverage = data['calibration_data'][param]
        
        # Plot the calibration curve
        ax.plot(std_levels * 2, actual_coverage, 'o-', linewidth=2, 
               label=f'Model 5 ({param})', color=param_colors[i], markersize=8)
        
        # Plot the ideal calibration line
        ax.plot([0, 1], [0, 1], '--', linewidth=2, color='gray', alpha=0.7, label='Ideal Calibration')
        
        # Shade the over/under confident regions
        ax.fill_between([0, 1], [0, 0], [0, 1], color='red', alpha=0.1, label='Under-confident')
        ax.fill_between([0, 1], [0, 1], [1, 1], color='blue', alpha=0.1, label='Over-confident')
        
        # Set labels and title
        ax.set_xlabel('Expected Coverage (2σ)', fontsize=14)
        ax.set_ylabel('Actual Coverage', fontsize=14)
        ax.set_title(f'Uncertainty Calibration: {param_names[i]}', fontsize=14, fontweight='bold')
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend (only for the first plot)
        if i == 0:
            ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 9. Training convergence and validation curves
def visualize_training_curves(save_path=None):
    """
    Create synthetic training and validation curves for different models.
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Synthetic training data
    epochs = np.arange(1, 151)
    
    # Supervised model (Model 1) convergence
    ax = axes[0, 0]
    
    # Synthetic learning curves (exponential decay with noise)
    np.random.seed(42)
    base_curve = 0.3 * np.exp(-epochs / 30) + 0.15
    train_noise = np.random.normal(0, 0.01, len(epochs))
    val_noise = np.random.normal(0, 0.015, len(epochs))
    
    train_loss = base_curve + train_noise
    val_loss = base_curve + 0.03 + val_noise
    
    # Early stopping point
    early_stop = 75
    val_min = np.argmin(val_loss[:early_stop]) + 1
    
    ax.plot(epochs, train_loss, '-', linewidth=2, label='Training Loss', color='#1f77b4')
    ax.plot(epochs, val_loss, '-', linewidth=2, label='Validation Loss', color='#ff7f0e')
    ax.axvline(x=val_min, linestyle='--', color='gray', linewidth=2, label='Early Stopping Point')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss (MSE)', fontsize=14)
    ax.set_title('Model 1 (Supervised) Training Convergence', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    # Self-supervised model (Model 2) convergence
    ax = axes[0, 1]
    
    # Different convergence pattern with augmentation
    base_curve = 0.4 * np.exp(-epochs / 40) + 0.2
    train_noise = np.random.normal(0, 0.02, len(epochs))
    val_noise = np.random.normal(0, 0.025, len(epochs))
    
    train_loss = base_curve + train_noise
    val_loss = base_curve + 0.02 + val_noise
    
    # Add the effect of reducing augmentation
    aug_reduction = np.minimum(epochs / 50, 1) * 0.05
    train_loss -= aug_reduction
    
    # Early stopping point
    early_stop = 100
    val_min = np.argmin(val_loss[:early_stop]) + 1
    
    ax.plot(epochs, train_loss, '-', linewidth=2, label='Training Loss', color='#1f77b4')
    ax.plot(epochs, val_loss, '-', linewidth=2, label='Validation Loss', color='#ff7f0e')
    ax.axvline(x=val_min, linestyle='--', color='gray', linewidth=2, label='Early Stopping Point')
    
    # Add annotation for augmentation reduction
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, edgecolor='gray')
    ax.annotate(
        "Augmentation ratio\ngradually reduced",
        xy=(50, train_loss[50] - 0.02), xytext=(70, train_loss[50] - 0.07),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        bbox=bbox_props, fontsize=12
    )
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss (Signal MSE)', fontsize=14)
    ax.set_title('Model 2 (Self-Supervised) Training Convergence', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    # VAE model (Model 5) convergence
    ax = axes[1, 0]
    
    # VAE with KL annealing
    base_curve = 0.5 * np.exp(-epochs / 50) + 0.25
    train_noise = np.random.normal(0, 0.02, len(epochs))
    val_noise = np.random.normal(0, 0.03, len(epochs))
    
    # KL annealing effect
    kl_effect = np.zeros_like(epochs, dtype=float)
    kl_effect[:20] = np.linspace(0, 0.1, 20)
    kl_effect[20:] = 0.1
    
    train_loss = base_curve + train_noise + kl_effect
    val_loss = base_curve + 0.03 + val_noise + kl_effect
    
    # KL divergence component
    kl_div = np.zeros_like(epochs, dtype=float)
    kl_div[:20] = np.linspace(0, 0.1, 20)
    kl_div[20:] = 0.1 * np.exp(-(epochs[20:] - 20) / 100)
    
    # Reconstruction component
    recon_loss = base_curve + train_noise
    
    # Early stopping point
    early_stop = 120
    val_min = np.argmin(val_loss[:early_stop]) + 1
    
    ax.plot(epochs, train_loss, '-', linewidth=2, label='Total Training Loss', color='#1f77b4')
    ax.plot(epochs, val_loss, '-', linewidth=2, label='Total Validation Loss', color='#ff7f0e')
    ax.plot(epochs, recon_loss, '--', linewidth=1.5, label='Reconstruction Loss', color='#2ca02c')
    ax.plot(epochs, kl_div, '--', linewidth=1.5, label='KL Divergence', color='#d62728')
    ax.axvline(x=val_min, linestyle='--', color='gray', linewidth=2, label='Early Stopping Point')
    
    # Add annotation for KL annealing
    ax.annotate(
        "KL annealing period",
        xy=(10, train_loss[10]), xytext=(30, train_loss[10] + 0.1),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        bbox=bbox_props, fontsize=12
    )
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Model 5 (VAE) Training Convergence', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    # Validation performance comparison
    ax = axes[1, 1]
    
    # Selected models
    models = ['Model 1', 'Model 2', 'Model 4', 'Model 5', 'Model 6']
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    # Validation rRMSE curves
    for i, model in enumerate(models):
        # Create synthetic convergence curve
        if model == 'Model 1':
            base = 0.25 * np.exp(-epochs / 40) + 0.2
            noise = np.random.normal(0, 0.01, len(epochs))
        elif model == 'Model 2':
            base = 0.23 * np.exp(-epochs / 45) + 0.19
            noise = np.random.normal(0, 0.01, len(epochs))
        elif model == 'Model 4':
            base = 0.28 * np.exp(-epochs / 50) + 0.21
            noise = np.random.normal(0, 0.015, len(epochs))
        elif model == 'Model 5':
            base = 0.26 * np.exp(-epochs / 60) + 0.2
            noise = np.random.normal(0, 0.012, len(epochs))
        elif model == 'Model 6':
            base = 0.22 * np.exp(-epochs / 40) + 0.17
            noise = np.random.normal(0, 0.008, len(epochs))
        
        val_rmse = base + noise
        
        # Plot curve
        ax.plot(epochs, val_rmse, '-', linewidth=2, label=model, color=colors[i])
        
        # Mark minimum
        min_idx = np.argmin(val_rmse) + 1
        ax.plot(min_idx, val_rmse[min_idx-1], 'o', markersize=8, color=colors[i])
    
    # Add reference line for NLLS baseline
    ax.axhline(y=0.336, linestyle='--', color='red', linewidth=2, label='NLLS Baseline')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Validation rRMSE', fontsize=14)
    ax.set_title('Validation Performance Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 10. Feature importance and ablation study
def visualize_ablation_study(save_path=None):
    """
    Create a visualization of feature importance and ablation study results.
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Ablation study for Model 4 (physical priors)
    ax = axes[0, 0]
    
    # Model configurations
    configs = [
        'No priors (Model 3)',
        'Range priors only',
        'Relationship priors only',
        'Smoothness priors only',
        'Range + Relationship',
        'All priors (Model 4)'
    ]
    
    # Synthetic performance data (rRMSE)
    overall_rrmse = [0.254, 0.243, 0.249, 0.246, 0.237, 0.232]
    tumor_rrmse = [0.198, 0.187, 0.190, 0.183, 0.176, 0.171]
    
    # Create grouped bar chart
    x = np.arange(len(configs))
    width = 0.35
    
    ax.bar(x - width/2, overall_rrmse, width, label='Overall rRMSE', color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, tumor_rrmse, width, label='Tumor rRMSE', color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add labels and legend
    ax.set_xlabel('Model Configuration', fontsize=14)
    ax.set_ylabel('Relative RMSE', fontsize=14)
    ax.set_title('Ablation Study: Impact of Physical Priors (Model 4)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotations
    best_idx = len(configs) - 1
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, edgecolor='gray')
    
    ax.annotate(
        "Combined priors yield\nbest performance",
        xy=(best_idx, tumor_rrmse[best_idx]), xytext=(best_idx - 1.5, tumor_rrmse[best_idx] - 0.02),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        bbox=bbox_props, fontsize=12
    )
    
    # Ablation study for Model 2 (data augmentation)
    ax = axes[0, 1]
    
    # Augmentation strategies
    strategies = [
        'No augmentation',
        'Uniform random params',
        'Tissue-specific params',
        'Adaptive noise levels',
        'Full augmentation'
    ]
    
    # Synthetic performance data
    rrmse_values = [0.242, 0.229, 0.213, 0.209, 0.202]
    
    # Create bar chart
    bars = ax.bar(strategies, rrmse_values, color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add labels
    ax.set_xlabel('Augmentation Strategy', fontsize=14)
    ax.set_ylabel('Overall rRMSE', fontsize=14)
    ax.set_title('Ablation Study: Impact of Data Augmentation (Model 2)', fontsize=16, fontweight='bold')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add improvement percentages
    baseline = rrmse_values[0]
    for i, bar in enumerate(bars):
        improvement = (baseline - rrmse_values[i]) / baseline * 100
        if improvement > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{improvement:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Ablation study for Model 6 (ensemble components)
    ax = axes[1, 0]
    
    # Ensemble configurations
    configs = [
        'Expert A (Model 1) only',
        'Expert B (Model 2) only',
        'Expert C (Model 5) only',
        'Simple averaging',
        'Weighted averaging',
        'Tissue-based gating',
        'Residual-based gating'
    ]
    
    # Synthetic performance data
    rrmse_values = [0.221, 0.202, 0.214, 0.212, 0.198, 0.193, 0.181]
    
    # Create bar chart
    bars = ax.bar(configs, rrmse_values, color='#d62728', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add labels
    ax.set_xlabel('Ensemble Configuration', fontsize=14)
    ax.set_ylabel('Overall rRMSE', fontsize=14)
    ax.set_title('Ablation Study: Impact of Ensemble Components (Model 6)', fontsize=16, fontweight='bold')
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add best expert baseline
    best_expert_idx = 1  # Expert B (Model 2)
    ax.axhline(y=rrmse_values[best_expert_idx], linestyle='--', color='#ff7f0e', 
              linewidth=2, label=f'Best Expert (Model 2)')
    ax.legend(loc='upper right', fontsize=12)
    
    # Add annotation
    ax.annotate(
        "Residual-based gating\nsignificantly outperforms\nbest individual expert",
        xy=(len(configs)-1, rrmse_values[-1]), 
        xytext=(len(configs)-3, rrmse_values[-1] - 0.03),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        bbox=bbox_props, fontsize=12
    )
    
    # Network architecture ablation
    ax = axes[1, 1]
    
    # Architecture variations
    architectures = [
        'Shallow MLP\n(2 layers)',
        'Standard MLP\n(5 layers)',
        'Deep MLP\n(8 layers)',
        'CNN\n(3 conv + 2 FC)',
        'Attention-based\n(transformer)'
    ]
    
    # Synthetic performance data
    overall_rrmse = [0.249, 0.221, 0.225, 0.234, 0.229]
    
    # Create bar chart
    bars = ax.bar(architectures, overall_rrmse, color='#9467bd', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add labels
    ax.set_xlabel('Network Architecture', fontsize=14)
    ax.set_ylabel('Overall rRMSE', fontsize=14)
    ax.set_title('Ablation Study: Impact of Network Architecture (Model 1)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight the best architecture
    best_idx = np.argmin(overall_rrmse)
    bars[best_idx].set_color('#ff7f0e')
    bars[best_idx].set_alpha(1.0)
    
    # Add annotation
    ax.annotate(
        "Optimal network depth\nbalances complexity\nand generalization",
        xy=(best_idx, overall_rrmse[best_idx]), 
        xytext=(best_idx + 1.5, overall_rrmse[best_idx] + 0.01),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        bbox=bbox_props, fontsize=12
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Main function to generate all visualizations
def generate_all_visualizations(output_dir='./figures/'):
    """
    Generate all visualizations and save them to the specified directory.
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample data
    print("Loading data...")
    data = load_sample_data()
    
    # Generate and save visualizations
    print("Generating visualizations...")
    
    print("1. Parameter Maps Comparison...")
    visualize_parameter_maps(data, os.path.join(output_dir, '1_parameter_maps.png'))
    
    print("2. Signal Decay Curves...")
    visualize_signal_decay(data, os.path.join(output_dir, '2_signal_decay.png'))
    
    print("3. Model Performance Comparison...")
    visualize_model_performance(data, os.path.join(output_dir, '3_model_performance.png'))
    
    print("4. Error Maps...")
    visualize_error_maps(data, os.path.join(output_dir, '4_error_maps.png'))
    
    print("5. Uncertainty Maps...")
    visualize_uncertainty_maps(data, os.path.join(output_dir, '5_uncertainty_maps.png'))
    
    print("6. Mixture-of-Experts Selection...")
    visualize_moe_selection(data, os.path.join(output_dir, '6_moe_selection.png'))
    
    print("7. B-value Performance...")
    visualize_b_value_performance(data, os.path.join(output_dir, '7_bvalue_performance.png'))
    
    print("8. Uncertainty Calibration...")
    visualize_uncertainty_calibration(data, os.path.join(output_dir, '8_uncertainty_calibration.png'))
    
    print("9. Training Curves...")
    visualize_training_curves(os.path.join(output_dir, '9_training_curves.png'))
    
    print("10. Ablation Study...")
    visualize_ablation_study(os.path.join(output_dir, '10_ablation_study.png'))
    
    print(f"All visualizations saved to {output_dir}")

# Run the script if executed directly
if __name__ == "__main__":
    generate_all_visualizations()