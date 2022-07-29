# TensionMap

TensionMap is a Python library for force inference and morphometrics for membrane-stained tissue images. Given a segmented, labelled cell array image, TensionMap can:

- Infer cell pressure and cell-cell junction tension using a python implementation of the VMSI method (see *Noll et al., 2020*).
- Measure 11 morphometric quantities for each cell.
- Visualise results of force inference.
- Output morphomtric and mechanical features for integration with gene expression data and further analysis.

## Installation

To be updated...

## Tutorials and examples

[Running pipeline on synthetic image and basic analysis](notebooks/synthetic_image.ipynb)
[Running pipeline on seqFISH mouse embryo dorsal region](notebooks/mouse_dorsal_seqfish_inference.ipynb)
[Integrated analysis of gene expression and mechanics/morphology for seqFISH mouse embryo dorsal region](notebooks/mouse_dorsal_seqfish_inference.ipynb)

## Quickstart

To run TensionMap on a segmented image, use the workflow below:

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
