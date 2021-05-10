# TensionMap
TensionMap is a Python library for end-to-end estimation of tension in epithelial cell images. It features a pipeline for cell topology deduction, as well as an implementation of the VMSI algorithm for tension inference, which can be evaluated on synthetically generated data. 

The three main features of TensionMap are:
- Synthetic data generator
- Segmentation and cell graph inference
- VMSI tension inference

As a rule of thumb, if the tension inference takes too much time (or fails to converge), try to decrease the size of your input image.
