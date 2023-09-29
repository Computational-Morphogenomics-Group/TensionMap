# TensionMap

TensionMap is a Python library for spatial mechano-transcriptomics allowing for image-based force inference, cell morphometrics and spatial omics data integration. Given an image-based spatial omics data set (seqFISH, MERFISH, cosMx, etc.) and accurate segmentation masks of cell contours TensionMap can:

- Infer cell pressure, tension at cell-cell junctions and cellular stress tensor using a python-based force inference method.
- Visualise results of force inference.
- Output cell and tissue level mechanical features for integration with spatial omics data.
- Perform mechano-transcriptomics analysis including:
  - tissue compartment boundary detection;
  - spatial ligand-receptor analysis;  
  - non-linear association models for detecting gene expression modules associated with cellular mechanics;
  - geoadditive structural equation models for detecting gene expression modules associated with cellular mechanics while controlling for spatial confounders.
    
## Reference

Adrien Hallou, Ruiyang He, Benjamin David Simons and Bianca Dumitrascu. A computational pipeline for spatial mechano-transcriptomics. bioRxiv 2023.08.03.551894; doi: https://doi.org/10.1101/2023.08.03.551894

## Table of Contents

1. [Installation](#installation)
2. [Tutorials and examples](#tutorials-and-examples)
3. [Quickstart](#quickstart)
4. [Notes and frequently encountered issues](#notes-and-frequently-encountered-issues)
5. [License](#license)

## Installation

First, clone the TensionMap repository to your local machine, where `<tensionmap_dir>` is a directory of your choice:

```
git clone --branch TensionMap-new https://github.com/Computational-Morphogenomics-Group/TensionMap.git <tensionmap_dir>
```

The requirements to run TensionMap alone can be found in `tensionmap-minimal.yml`. Create a conda environment with the required dependencies using:

```
conda env create -f tensionmap-minimal.yml -n tensionmap
```

To run the other post force inference analysis scripts found in the Tutorials section, create a conda environment using `tensionmap-full.yml` instead. This installs libraries for integrated analysis with gene expression data, such as `scanpy` and `Phenograph`.

You can activate the environment using `conda activate tensionmap`. To use TensionMap in a python script, first add the `src` directory to `sys.path`, then import as normal:

```
import sys
sys.path.append('<tensionmap_dir>/src')

from src.VMSI import *
```

### Optional: Matlab `fmincon` optimiser

TensionMap can use optimisers from the Python implementation of the `NLopt` library or Matlab's `fmincon`. To enable `fmincon` for TensionMap, a licensed installation of Matlab is needed, and the Matlab Engine API for Python must be installed. Full instructions can be found [here](https://mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

To install the Matlab Engine API, first identify the folder where Matlab is installed. This can be found by opening Matlab and typing `matlabroot` into the console. 

Next, execute the following commands, where `<matlabroot>` is the Matlab install directory.

For Linux/macOS:

```
cd "<matlabroot>/extern/engines/python"
python setup.py install
```

For Windows:

```
cd "<matlabroot>\extern\engines\python"
python setup.py install
```

Note that you may need to use `python3` instead of `python` if you have both Python 2 and Python 3 installed.

## Tutorials and examples

#### Basic tutorial
[Running pipeline on synthetic image and basic analysis](notebooks/synthetic_image.ipynb) <br />

#### Reproducing further mechano-transcriptomics analysis from the manuscript
[Run TensionMap on mouse embryo seqFISH data](notebooks/00_run_tensionmap.ipynb) <br />
[Characterising mechanical properties of tissue boundaries](notebooks/01_biophysical_analysis.ipynb) <br />
[Analyse imputed whole-transcriptome single-cell expression data](notebooks/02_sc_analysis.ipynb) <br />
[Identifying putative ligand-receptor signalling across boundaries](notebooks/03_lr_analysis.ipynb) <br />
[Identifying nonlinear association patterns between gene expression and cellular mechanics](notebooks/04_nonlinear_schot.ipynb) <br />
[Testing for associations between gene expression and cellular mechanics using a spatial regression model](notebooks/05_spatial_regression.ipynb) <br />

## Quickstart

To run TensionMap on a segmented image, use the workflow below:

Load required packages and add TensionMap to sys.path, where `'tensionmap_path'` is where TensionMap was downloaded:

```
import sys
sys.path.append('tensionmap_path')
from src.VMSI import *
import skimage
from skimage import io
```

First, load the image:

```
img = skimage.io.imread('img_path')
```

If boundaries between cells are not delineated (i.e. cell labels are touching), find cell boundaries:

```
img = skimage.segmentation.find_boundaries(img, mode='subpixel')
img = skimage.measure.label(1-img)
```

Next, run the force inference and morphometrics:

```
vmsi_model = run_VMSI(img)
```

Results from the resulting model can be plotted using the `plot()` method, or output as a Pandas dataframe or .csv file using the `output_results()` method:

```
vmsi_model.plot(['tension','pressure', 'stress'], img)
results = vmsi_model.output_results()
```

## Notes and frequently encountered issues

- Inference is dependent on a high-quality segmentation; a common issue that can result in errors is segmentation artifacts resulting from manual segmentations. 

- Image tiling may lead to unexpected results and is not recommended for images which exhibit significant spatial anisotropy in cell morphology.

## License 

- This project is licensed under the terms of the MIT License.
