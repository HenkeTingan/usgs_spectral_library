# USGS Spectral Library Analysis

This project processes and analyzes mineral spectral data from the USGS Spectral Library Version 7. It generates both raw spectra and derivative spectra plots for various minerals.

## Minerals Analyzed
- Smectite
- Illite
- Chlorite
- Kaolinite
- Dolomite
- Calcite
- Quartz
- Feldspar

## Features
- Processes spectral data from USGS Spectral Library
- Generates raw reflectance spectra plots
- Calculates and plots derivative spectra
- Saves processed data in CSV format
- Creates combined plots for all minerals

## Requirements
- Python 3.x
- numpy
- pandas
- matplotlib
- scipy

## Installation
1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
Run the script to process the spectral data:
```bash
python process_spectral_data.py
```

This will generate:
- Individual mineral spectra plots (*_spectra.png)
- Derivative spectra plots (*_derivative.png)
- Combined spectra plots (combined_spectra.png, combined_derivative.png)
- Processed data files (*_processed.csv)

## Data Source
The spectral data is sourced from the USGS Spectral Library Version 7, specifically from the ChapterM_Minerals directory.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 