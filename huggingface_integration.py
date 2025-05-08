import os
import numpy as np
import pandas as pd
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image
import matplotlib.pyplot as plt

class SpectralAnalyzer:
    def __init__(self, model_name="microsoft/resnet-50"):
        """
        Initialize the spectral analyzer with a Hugging Face model.
        
        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
    def prepare_spectral_data(self, wavelengths, reflectance):
        """
        Convert spectral data to an image format suitable for the model.
        
        Args:
            wavelengths (np.array): Array of wavelength values
            reflectance (np.array): Array of reflectance values
            
        Returns:
            PIL.Image: Image representation of the spectral data
        """
        # Create a figure and plot the spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, reflectance)
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Reflectance')
        plt.grid(True)
        
        # Save the plot to a temporary file
        temp_file = 'temp_spectrum.png'
        plt.savefig(temp_file)
        plt.close()
        
        # Load the image
        image = Image.open(temp_file)
        
        # Clean up the temporary file
        os.remove(temp_file)
        
        return image
    
    def analyze_spectrum(self, wavelengths, reflectance):
        """
        Analyze a spectrum using the Hugging Face model.
        
        Args:
            wavelengths (np.array): Array of wavelength values
            reflectance (np.array): Array of reflectance values
            
        Returns:
            dict: Model predictions and confidence scores
        """
        # Prepare the spectral data
        image = self.prepare_spectral_data(wavelengths, reflectance)
        
        # Prepare the input for the model
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.softmax(dim=1)
        
        # Convert predictions to a more readable format
        results = {
            'predictions': predictions.cpu().numpy(),
            'confidence': float(predictions.max().cpu().numpy())
        }
        
        return results

def main():
    # Example usage
    analyzer = SpectralAnalyzer()
    
    # Example spectral data (replace with your actual data)
    wavelengths = np.linspace(0.35, 2.5, 2151)
    reflectance = np.random.random(2151)  # Replace with actual reflectance data
    
    # Analyze the spectrum
    results = analyzer.analyze_spectrum(wavelengths, reflectance)
    print("Analysis results:", results)

if __name__ == "__main__":
    main() 