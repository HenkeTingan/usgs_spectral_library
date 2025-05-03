import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

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

def plot_spectra(spectra_data, wavelengths, title, output_file, derivative=False):
    """Plot multiple spectra on the same graph."""
    try:
        plt.figure(figsize=(12, 6))
        for mineral, reflectance in spectra_data.items():
            if derivative:
                y_data = calculate_derivative(wavelengths, reflectance)
                if y_data is None:
                    continue
            else:
                y_data = reflectance
            plt.plot(wavelengths, y_data, label=os.path.basename(mineral))
        
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('First Derivative' if derivative else 'Reflectance')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved plot to {output_file}")
    except Exception as e:
        print(f"Error plotting spectra: {str(e)}")

def find_mineral_files(mineral_name, base_path):
    """Find all files related to a specific mineral."""
    mineral_files = []
    try:
        # Look for files containing the mineral name (case insensitive)
        # and ending with specific extensions
        for root, _, files in os.walk(base_path):
            for file in files:
                # Look for ASDFR files (standard resolution) with AREF (absolute reflectance)
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
    
    # Check if directory exists
    if not os.path.exists(base_path):
        print(f"Directory not found: {base_path}")
        return mineral_data
    
    # Find all relevant files
    mineral_files = find_mineral_files(mineral_name, base_path)
    print(f"Found {len(mineral_files)} files for {mineral_name}:")
    for file in mineral_files:
        print(f"  {os.path.basename(file)}")
    
    # Process each file
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
            
            # Plot individual mineral spectra
            plot_spectra(
                mineral_data,
                wavelengths,
                f'{mineral.capitalize()} Spectra',
                f'{mineral}_spectra.png'
            )
            
            # Plot derivative spectra
            plot_spectra(
                mineral_data,
                wavelengths,
                f'{mineral.capitalize()} Derivative Spectra',
                f'{mineral}_derivative.png',
                derivative=True
            )
            
            # Save processed data
            for sample_name, reflectance in mineral_data.items():
                output_name = f'{mineral}_{os.path.basename(sample_name).replace(".txt", "")}_processed.csv'
                try:
                    df = pd.DataFrame({
                        'wavelength': wavelengths,
                        'reflectance': reflectance,
                        'derivative': calculate_derivative(wavelengths, reflectance)
                    })
                    df.to_csv(output_name, index=False)
                    print(f"Saved processed data to {output_name}")
                except Exception as e:
                    print(f"Error saving processed data: {str(e)}")
        else:
            print(f"No data found for {mineral}")
    
    # Plot all minerals together
    if all_mineral_data:
        # Combine first spectrum of each mineral
        combined_data = {}
        for mineral, spectra in all_mineral_data.items():
            if spectra:
                first_spectrum = list(spectra.items())[0]
                combined_data[mineral] = first_spectrum[1]
        
        # Plot combined spectra
        plot_spectra(
            combined_data,
            wavelengths,
            'Combined Mineral Spectra',
            'combined_spectra.png'
        )
        
        # Plot combined derivative spectra
        plot_spectra(
            combined_data,
            wavelengths,
            'Combined Mineral Derivative Spectra',
            'combined_derivative.png',
            derivative=True
        )

if __name__ == "__main__":
    main() 