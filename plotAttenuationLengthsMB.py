from NuRadioReco.utilities import units
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_attenuation_length(freqs, apply_correction=True, R=0.82):
    """
    Calculates the attenuation length in ice as a function of frequency,
    with optional corrections for reflectivity at the ice bottom.

    Parameters:
    freqs (np.ndarray): Array of frequencies.
    apply_correction (bool): Whether to apply the reflectivity correction.
    R (float): Reflectivity of ice bottom.

    Returns:
    np.ndarray: Calculated attenuation lengths.
    """
    # Constants
    d_ice = 576 * units.m  # Depth of ice at Moore's Bay

    # Initial linear model for attenuation length
    att_length = 460 * units.m - (180 * units.m / units.GHz) * freqs

    if apply_correction:
        # Correction factor due to reflectivity (R) not being 1
        correction_factor = (1 + (att_length / (2 * d_ice)) * np.log(R)) ** -1
        att_length *= correction_factor

    # NOTE: Temperature dependence is currently ignored but may be needed later.

    # Ensure attenuation length is a positive value
    att_length[att_length <= 0] = 1 * units.m
    return att_length

def plot_attenuation_comparison():
    """
    Creates comparison plots of attenuation lengths with and without correction.
    """
    # Create output directory
    os.makedirs('plots/attenuation', exist_ok=True)
    
    # Define frequency range from 0 to 500 MHz
    frequencies = np.arange(0, 501, 1) * units.MHz
    
    # Calculate different scenarios
    att_no_correction = calculate_attenuation_length(frequencies, apply_correction=False)
    att_with_correction = calculate_attenuation_length(frequencies, apply_correction=True, R=0.82)
    
    # Plot 1: Attenuation lengths without and with correction factor
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frequencies / units.MHz, att_no_correction / units.m, 
            label='Without correction factor', linewidth=2)
    ax.plot(frequencies / units.MHz, att_with_correction / units.m, 
            label='With correction factor (R=0.82)', linewidth=2)
    
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Attenuation Length [m]")
    ax.set_title("Ice Attenuation Length at Moore's Bay - Correction Factor Comparison")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 500)
    ax.set_ylim(bottom=0)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/attenuation/attenuation_correction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Attenuation lengths multiplied by R values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frequencies / units.MHz, (att_no_correction * 1.0) / units.m, 
            label='Without correction × R=1', linewidth=2)
    ax.plot(frequencies / units.MHz, (att_no_correction * 0.82) / units.m, 
            label='Without correction × R=0.82', linewidth=2)
    ax.plot(frequencies / units.MHz, (att_with_correction * 1.0) / units.m, 
            label='With correction × R=1', linewidth=2, linestyle='--')
    ax.plot(frequencies / units.MHz, (att_with_correction * 0.82) / units.m, 
            label='With correction × R=0.82', linewidth=2, linestyle='--')
    
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Attenuation Length × R [m]")
    ax.set_title("Ice Attenuation Length at Moore's Bay - Multiplied by Reflectivity")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 500)
    ax.set_ylim(bottom=0)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/attenuation/attenuation_multiplied_by_R.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Combined plot with all scenarios
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Subplot 1: Original comparison
    ax1.plot(frequencies / units.MHz, att_no_correction / units.m, 
             label='Without correction factor', linewidth=2, color='blue')
    ax1.plot(frequencies / units.MHz, att_with_correction / units.m, 
             label='With correction factor (R=0.82)', linewidth=2, color='red')
    
    ax1.set_xlabel("Frequency [MHz]")
    ax1.set_ylabel("Attenuation Length [m]")
    ax1.set_title("Attenuation Length Comparison")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_xlim(0, 500)
    ax1.set_ylim(bottom=0)
    ax1.legend()
    
    # Subplot 2: Multiplied by R values
    ax2.plot(frequencies / units.MHz, (att_no_correction * 1.0) / units.m, 
             label='Without correction × R=1', linewidth=2, color='lightblue')
    ax2.plot(frequencies / units.MHz, (att_no_correction * 0.82) / units.m, 
             label='Without correction × R=0.82', linewidth=2, color='blue')
    ax2.plot(frequencies / units.MHz, (att_with_correction * 1.0) / units.m, 
             label='With correction × R=1', linewidth=2, linestyle='--', color='lightcoral')
    ax2.plot(frequencies / units.MHz, (att_with_correction * 0.82) / units.m, 
             label='With correction × R=0.82', linewidth=2, linestyle='--', color='red')
    
    ax2.set_xlabel("Frequency [MHz]")
    ax2.set_ylabel("Attenuation Length × R [m]")
    ax2.set_title("Attenuation Length Multiplied by Reflectivity")
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_xlim(0, 500)
    ax2.set_ylim(bottom=0)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('plots/attenuation/attenuation_combined_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_attenuation_length(freqs, att_length):
    """
    Plots the attenuation length as a function of frequency.
    (Original function kept for compatibility)

    Parameters:
    freqs (np.ndarray): Array of frequencies.
    att_length (np.ndarray): Array of attenuation lengths.
    """
    fig, ax = plt.subplots()

    # Plotting without units for clarity
    ax.plot(freqs / units.MHz, att_length / units.m)

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Attenuation Length [m]")
    ax.set_title("Ice Attenuation Length at Moore's Bay")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 500)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Create all the requested plots
    plot_attenuation_comparison()

    def plot_normalized_attenuation_factor():
        """
        Plots the normalized attenuation factor as a function of frequency.
        Formula: exp(-2 * 576 / (cos(zenith) * L_att))
        Zenith: 45 degrees
        Normalized such that 100 MHz = 1.0
        """
        # Create output directory
        os.makedirs('plots/attenuation', exist_ok=True)
        
        # Define frequency range from 0 to 500 MHz
        frequencies = np.arange(0, 501, 1) * units.MHz
        
        # Calculate attenuation length with correction
        att_length = calculate_attenuation_length(frequencies, apply_correction=True, R=0.82)
        
        # Constants
        d_ice = 576 * units.m
        zenith_deg = 45
        zenith_rad = np.deg2rad(zenith_deg)
        
        # Calculate the factor
        # Path length for reflection: 2 * d_ice / cos(zenith)
        # Formula: exp(-2 * 576 / (cos(zenith) * L_att))
        exponent = -2 * d_ice / (np.cos(zenith_rad) * att_length)
        factor = np.exp(exponent)
        
        # Normalize at 100 MHz
        # Find index for 100 MHz. Since step is 1 MHz and start is 0, index 100 is 100 MHz.
        idx_100MHz = 100
        norm_value = factor[idx_100MHz]
        
        normalized_factor = factor / norm_value
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frequencies / units.MHz, normalized_factor, linewidth=2, label='Normalized Attenuation Factor')
        
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Normalized Factor (1.0 at 100 MHz)")
        ax.set_title("Normalized Attenuation Factor vs Frequency")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(0, 500)
        
        # Add text for zenith
        ax.text(0.05, 0.95, f"Zenith: {zenith_deg}°", transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                
        plt.tight_layout()
        plt.savefig('plots/attenuation/normalized_attenuation_factor.png', dpi=300, bbox_inches='tight')
        plt.show()

    plot_normalized_attenuation_factor()
    
    # Original functionality (optional)
    # frequencies = np.arange(0, 501, 1) * units.MHz
    # attenuation_lengths = calculate_attenuation_length(frequencies)
    # plot_attenuation_length(frequencies, attenuation_lengths)