import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from scipy import statsd


def analyze_sequence_distribution(
    english_tokens: List[List[int]],
    hindi_tokens: List[List[int]],
    std_multiplier: float = 2.0
) -> Tuple[int, Dict]:
    """
    Analyze sequence length distribution and recommend optimal length based on
    mean and standard deviation. Uses the empirical rule to capture most data.
    
    Args:
        english_tokens: List of English token sequences
        hindi_tokens: List of Hindi token sequences
        std_multiplier: Number of standard deviations above mean (default: 2.0)
        
    Returns:
        Tuple containing:
        - recommended sequence length
        - dictionary with statistical metrics
    """
    if len(english_tokens) != len(hindi_tokens):
        raise ValueError("English and Hindi token lists must have the same length")
    
    # Get sequence lengths, taking max of Hindi and English for each pair
    sequence_lengths = [
        max(len(eng), len(hin)) 
        for eng, hin in zip(english_tokens, hindi_tokens)
    ]
    
    # Calculate basic statistics
    mean_length = np.mean(sequence_lengths)
    std_length = np.std(sequence_lengths)
    
    # Calculate recommended sequence length (mean + n*std)
    recommended_length = int(np.ceil(mean_length + std_multiplier * std_length))
    
    # Calculate percentiles
    percentiles = np.percentile(sequence_lengths, [25, 50, 75, 90, 95, 99])
    
    # Calculate how many sequences would be preserved
    preserved_sequences = sum(1 for length in sequence_lengths if length <= recommended_length)
    preservation_ratio = preserved_sequences / len(sequence_lengths)
    
    metrics = {
        'mean_length': mean_length,
        'std_length': std_length,
        'recommended_length': recommended_length,
        'total_sequences': len(sequence_lengths),
        'preserved_sequences': preserved_sequences,
        'preservation_ratio': preservation_ratio,
        'percentiles': {
            'p25': percentiles[0],
            'p50': percentiles[1],
            'p75': percentiles[2],
            'p90': percentiles[3],
            'p95': percentiles[4],
            'p99': percentiles[5]
        },
        'sequence_lengths': sequence_lengths
    }
    
    return recommended_length, metrics

def plot_length_distribution(metrics: Dict, save_path: str = None):
    """
    Plot the distribution of sequence lengths with key statistics.
    
    Args:
        metrics: Dictionary containing statistical metrics
        save_path: Optional path to save the plot
    """
    sequence_lengths = metrics['sequence_lengths']
    recommended_length = metrics['recommended_length']
    mean_length = metrics['mean_length']
    std_length = metrics['std_length']
    
    plt.figure(figsize=(12, 6))
    
    # Create histogram with density plot
    counts, bins, _ = plt.hist(sequence_lengths, bins=30, density=True, 
                              alpha=0.6, color='skyblue', label='Actual Distribution')
    
    # Add kernel density estimation
    kde = stats.gaussian_kde(sequence_lengths)
    x_range = np.linspace(min(sequence_lengths), max(sequence_lengths), 200)
    plt.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
    
    # Add vertical lines for key metrics
    plt.axvline(mean_length, color='green', linestyle='--', 
                label=f'Mean ({mean_length:.1f})')
    plt.axvline(recommended_length, color='red', linestyle='-', 
                label=f'Recommended ({recommended_length})')
    
    # Add text box with statistics
    stats_text = (
        f"Mean: {mean_length:.1f}\n"
        f"Std: {std_length:.1f}\n"
        f"Recommended: {recommended_length}\n"
        f"Preservation: {metrics['preservation_ratio']*100:.1f}%\n"
        f"95th percentile: {metrics['percentiles']['p95']:.1f}"
    )
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()