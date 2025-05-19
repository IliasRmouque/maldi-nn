from maldi_nn.spectrum import *
import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser
from concurrent.futures import ProcessPoolExecutor
import os



def process_spectrum(args):
    """Process a single spectrum"""
    mz, intensity, preprocessor, x, y = args
    spectrum = SpectrumObject(mz=mz, intensity=intensity)
    spectrum_preprocessed = preprocessor(spectrum)
    return spectrum_preprocessed, x , y

def create_preprocessor():
    """Create the preprocessor pipeline"""
    return SequentialPreprocessor(
        VarStabilizer(method="sqrt"),
        Smoother(halfwindow=10),
        BaselineCorrecter(method="SNIP", snip_n_iter=20),
        Normalizer(sum=1),
        PeakFilter(max_number=200),
    )

if __name__ == "__main__":
    existing_images = "/media/ilias/Crucial X9/MALDI/existing_images.csv"
    df = pd.read_csv(existing_images)
    imzmls = [i.split('\'')[1] for i in df["imzml"]]
    
    max_workers = os.cpu_count() - 2  # Leave one CPU for system processes
    
    for path in imzmls:
        output_name = "../data/"+ path.split("/")[-1].split(".")[0]

        if not os.path.exists(path):
            print(f"File {path} does not exist. Skipping...")
            continue
        if os.path.exists(output_name + ".npz"):
            print(f"File {path.split('.')[0]}.npz already exists. Skipping...")
            continue
        print(f"Processing {os.path.basename(path)}...")
        image = ImzMLParser(path, include_spectra_metadata=None)
        
        # Prepare inputs for parallel processing
        spectrum_data = []
        preprocessor = create_preprocessor()
        
        for i, (x, y, z) in enumerate(image.coordinates):
            mz, intensity = image.getspectrum(i)
            spectrum_data.append((mz, intensity, preprocessor, x, y))
        
        # Process spectra in parallel
        mzs = []
        spectrums = []
        xs = []
        ys = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_spectrum, spectrum_data))
            
        for spectrum, x, y in results:
            mzs.append(spectrum.mz)
            spectrums.append(spectrum.intensity)
            xs.append(x)
            ys.append(y)

        
        # Save the processed data
        np.savez_compressed(output_name, mzs=np.array(mzs), spectrums=np.array(spectrums), x=np.array(xs), y=np.array(ys))
    
 
