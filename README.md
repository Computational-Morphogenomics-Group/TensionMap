# TensionMap

TensionMap is a Python library for force inference and morphometrics for membrane-stained tissue images. Given a segmented, labelled cell array image, TensionMap can:

- Infer cell pressure and cell-cell junction tension using a python implementation of the VMSI method (see *Noll et al., 2020*).
- Measure 11 morphometric quantities for each cell.
- Visualise results of force inference.
- Output morphomtric and mechanical features for integration with gene expression data and further analysis.

## Installation

First, clone the Tensionmap repository to your local machine, where `tensionmap_dir` is a directory of your choice:

```
git clone --branch TensionMap-new https://github.com/Computational-Morphogenomics-Group/TensionMap.git <tensionmap_dir>
```

The requirements to run TensionMap alone can be found in `tensionmap_minimal.yml`. Create a conda environment with the required dependencies using

```
conda env create -f tensionmap_minimal.yml -n tensionmap
```

You can activate the environment using `conda activate tensionmap`. To use TensionMap in a python script, first add the `src` directory to `sys.path`, then import as normal:

```
import sys
sys.path.append('<tensionmap_dir>/src')

from src.VMSI import *
```

### Optional: Matlab `fmincon` optimiser

TensionMap can utilise optimisers from the Python implementation of the `NLopt` library or Matlab's `fmincon`, used in the original manuscript of *Noll et al., 2020*. To enable `fmincon` for TensionMap, a licensed installation of Matlab is needed, and the Matlab Engine API for Python must be installed. Full instructions for this installation can be found [here](https://mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

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

[Running pipeline on synthetic image and basic analysis](notebooks/synthetic_image.ipynb) <br />
[Running pipeline on seqFISH mouse embryo dorsal region](notebooks/mouse_dorsal_seqfish_inference.ipynb) <br />
[Integrated analysis of gene expression and mechanics/morphology for seqFISH mouse embryo dorsal region](notebooks/mouse_dorsal_seqfish_analysis.ipynb)

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

## Notes/Frequently encountered issues

- Inference is dependent on a high-quality segmentation; a common issue that can result in errors is segmentation artifacts resulting from manual segmentations. 

- Image tiling may lead to unexpected results and is not recommended for images which exhibit significant spatial anisotropy in cell morphology.
