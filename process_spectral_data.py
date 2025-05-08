import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from geochemical_plotter import analyze_geochemical_data

def read_wavelength_file(base_path):
    """Read the wavelength values from the standard wavelength file."""
    wavelength_file = os.path.join(base_path, '..', 'splib07b_Wavelengths_ASDFR_0.35-2.5microns_2151ch.txt')
    try:
        with open(wavelength_file, 'r') as f:
            lines = f.readlines()
            # Skip header line
            wavelengths = [float(line.strip()) for line in lines[1:] if line.strip()]
        return np.array(wavelengths)
    except Exception as e:
        print(f"Error reading wavelength file: {str(e)}")
        return None

def read_spectral_file(file_path):
    """Read a spectral data file and return reflectance data."""
    try:
        print(f"Reading file: {file_path}")
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Skip header line
            reflectance = [float(line.strip()) for line in lines[1:] if line.strip()]
        return np.array(reflectance)
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def calculate_derivative(wavelength, reflectance, window_length=5, polyorder=2):
    """Calculate the first derivative of the reflectance spectrum."""
    try:
        return signal.savgol_filter(reflectance, window_length, polyorder, deriv=1)
    except Exception as e:
        print(f"Error calculating derivative: {str(e)}")
        return None

def plot_swir_spectra(spectra_data, wavelengths, title, output_file, derivative=False):
    """Plot multiple spectra focusing on the SWIR region (1.4-2.5 μm)."""
    try:
        # Filter wavelengths for SWIR region
        swir_mask = (wavelengths >= 1.4) & (wavelengths <= 2.5)
        swir_wavelengths = wavelengths[swir_mask]
        
        plt.figure(figsize=(12, 6))
        for mineral, reflectance in spectra_data.items():
            if derivative:
                y_data = calculate_derivative(wavelengths, reflectance)
                if y_data is None:
                    continue
            else:
                y_data = reflectance
            
            # Filter reflectance data for SWIR region
            swir_reflectance = y_data[swir_mask]
            
            # Get mineral name from file path
            mineral_name = os.path.basename(mineral).split('_')[0]
            plt.plot(swir_wavelengths, swir_reflectance, label=mineral_name)
        
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('First Derivative' if derivative else 'Reflectance')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved SWIR plot to {output_file}")
    except Exception as e:
        print(f"Error plotting SWIR spectra: {str(e)}")

def find_mineral_files(mineral_name, base_path):
    """Find all files related to a specific mineral."""
    mineral_files = []
    try:
        for root, _, files in os.walk(base_path):
            for file in files:
                if (mineral_name.lower() in file.lower() and 
                    file.endswith('.txt') and 
                    'ASDFR' in file and 
                    'AREF' in file):
                    mineral_files.append(os.path.join(root, file))
        return mineral_files
    except Exception as e:
        print(f"Error finding mineral files: {str(e)}")
        return []

def process_mineral_data(mineral_name, base_path, wavelengths):
    """Process all spectral data files for a given mineral."""
    mineral_data = {}
    print(f"\nLooking for {mineral_name} in {base_path}")
    
    if not os.path.exists(base_path):
        print(f"Directory not found: {base_path}")
        return mineral_data
    
    mineral_files = find_mineral_files(mineral_name, base_path)
    print(f"Found {len(mineral_files)} files for {mineral_name}:")
    for file in mineral_files:
        print(f"  {os.path.basename(file)}")
    
    for file_path in mineral_files:
        reflectance = read_spectral_file(file_path)
        if reflectance is not None and len(reflectance) > 0:
            mineral_data[file_path] = reflectance
            print(f"Successfully processed {os.path.basename(file_path)}")
        else:
            print(f"No valid data found in {os.path.basename(file_path)}")
    
    return mineral_data

def main():
    # Define minerals of interest
    minerals = [
        'smectite',
        'illite',
        'chlorite',
        'kaolinite',
        'dolomite',
        'calcite',
        'quartz',
        'feldspar'
    ]
    
    # Base path to the spectral data
    base_path = os.path.join('ASCIIdata', 'ASCIIdata_splib07b', 'ChapterM_Minerals')
    print(f"Base path: {base_path}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Read wavelength values
    wavelengths = read_wavelength_file(base_path)
    if wavelengths is None:
        print("Failed to read wavelength file. Exiting.")
        return
    
    print(f"Successfully read {len(wavelengths)} wavelength values")
    
    # Process and plot data for each mineral
    all_mineral_data = {}
    for mineral in minerals:
        print(f"\nProcessing {mineral}...")
        mineral_data = process_mineral_data(mineral, base_path, wavelengths)
        
        if mineral_data:
            print(f"Found {len(mineral_data)} samples for {mineral}")
            all_mineral_data[mineral] = mineral_data
            
            # Plot individual mineral spectra (SWIR region only)
            plot_swir_spectra(
                mineral_data,
                wavelengths,
                f'{mineral.capitalize()} SWIR Spectra',
                f'{mineral}_swir_spectra.png'
            )
            
            # Plot individual mineral derivative spectra (SWIR region only)
            plot_swir_spectra(
                mineral_data,
                wavelengths,
                f'{mineral.capitalize()} SWIR Derivative Spectra',
                f'{mineral}_swir_derivative.png',
                derivative=True
            )
    
    # Plot all minerals together
    if all_mineral_data:
        # Combine first spectrum of each mineral
        combined_data = {}
        for mineral, spectra in all_mineral_data.items():
            if spectra:
                first_spectrum = list(spectra.items())[0]
                combined_data[mineral] = first_spectrum[1]
        
        # Plot combined spectra (SWIR region only)
        plot_swir_spectra(
            combined_data,
            wavelengths,
            'Combined Mineral SWIR Spectra',
            'combined_swir_spectra.png'
        )
        
        # Plot combined derivative spectra (SWIR region only)
        plot_swir_spectra(
            combined_data,
            wavelengths,
            'Combined Mineral SWIR Derivative Spectra',
            'combined_swir_derivative.png',
            derivative=True
        )

    # Replace with your Excel file path
    analyze_geochemical_data("your_data.xlsx")

if __name__ == "__main__":
    main() 