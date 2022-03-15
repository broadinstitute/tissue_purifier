---
<div align="center">    
 
# Tissue Purifier: A library per the analysis of biological tissues   

[![Build Status](https://github.com/broadinstitute/millipede/workflows/CI/badge.svg)](https://github.com/broadinstitute/millipede/actions)
[![Documentation Status](https://readthedocs.org/projects/tissue_purifier/badge/?version=latest)](https://tissue-purifier.readthedocs.io/en/latest/?badge=latest)
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://NOT YET AVAILABLE.com/XXX)
</div>

## Description  
*Tissue Purifier* is a library for the analysis of biological tissues in Python.
It is built on [PyTorch](https://pytorch.org/), [Pyro](https://pyro.ai/) and 
[Anndata](https://anndata.readthedocs.io/en/latest/).

Spatially resolved transcriptomic technologies (such as 
[SlideSeq](https://pubmed.ncbi.nlm.nih.gov/30923225/),
[MerFish](https://www.sciencedirect.com/science/article/abs/pii/S0076687916001324), 
[SmFish](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6101419/),
[BaristaSeq](https://academic.oup.com/nar/article/46/4/e22/4668654), 
[ExSeq](https://pubmed.ncbi.nlm.nih.gov/33509999/), 
[STARMap](https://pubmed.ncbi.nlm.nih.gov/29930089/)
and others) allow measuring gene expression with spatial resolution. 
Deconvolution methods and/or analysis of marker-genes, can be used to assign
a discrete cell-type (such as Macrophage, B-Cells, ...) to each cell. 

This type of data can be nicely organized into anndata objects, which are data-structure 
specifically designed for transcriptomic data. 
Each anndata object contains a list of all the cells in a tissue together with (at the minimum):
1. the gene expression profile 
2. the cell-type label
3. the spatial coordinates (either in 2D or 3D) 

This rich data can unlock interesting scientific discoveries, but it is difficult to analyze.
Here is where *Tissue Purifier* comes in.

**In short, tissues are converted into images and cropped into overlapping patches.
Semantic features are associated to each patch via self supervised learning (ssl). 
The learned features are then used in downstream tasks (such as differential gene expression analysis).**

What's appealing about this approach is that it is unbiased, meaning that the researcher does not need to know 
*a priori* which features are important. Given enough data and a sufficiently large neural network this approach
should be able to extract semantic features useful in solving the downstream task. Negative results, 
meaning that the extracted features are useless in solving downstream tasks, are also interesting because they suggest 
that the task at hand *can not* be solved based on tissue structure alone (cell-type label and spatial coordinates).

## Typical workflow

A typical workflow consists of 3 steps:

1. Multiple anndata objects (corresponding to multiple tissues in possibly a diverse set of conditions) 
   are converted to (sparse) images. These images are cropped into overlapping patches of a characteristic length 
   (see [Documentation](https://tissue_purifier.readthedocs.io/en/latest) for a detailed discussion about the choice of 
   this length scale) and are fed into a ssl framework. 
   Importantly, in this step the model has no access to the gene expression profile. 
   It only uses the cell-type labels together with the spatial coordinates to create a multi-channel image 
   (in which each channel encodes the density of a specific cell-type). Therefore, the model can only leverage the 
   cells co-expression (sometimes referred to as a micro-environment) as a learning signal.
   See [notebook1](https://github.com/broadinstitute/tissue_purifier/blob/main/notebooks/notebook1.ipynb)  

2. Once a model is trained, any new anndata object can be processed. 
   As described above, the anndata object is transformed into a sparse image and cropped into 
   overlapping patches. Semantic features are associated to each patch and then transferred 
   to the cells belonging to the patch. Ultimately each cell in the anndata object is associated with a new set of 
   annotations which describe the local micro-environment of that cell. 
   This steps can be repeated multiple times (once for each of the models which have been trained) to compare 
   the quality of the features generated by multiple ssl frameworks.
   See [notebook2](https://github.com/broadinstitute/tissue_purifier/blob/main/notebooks/notebook2.ipynb)

3. Finally, we evaluate the quality of the features generated by each ssl framework. 
   To this end we use the annotations stored on the anndata object to predict the gene expression profile 
   conditioned on the cell-type. We compare with multiple baselines to show that the ssl features are biological
   informative. 
   See [notebook3](https://github.com/broadinstitute/tissue_purifier/blob/main/notebooks/notebook3.ipynb)  
   
## Why image-based self supervised learning?
Spatial transcriptomic data is a type of tabular data and could be analyzed without converting it to images. 
However, image-based approaches offer two remarkable advantages:
1. We can leverage state-of-the-art approaches which are continuously developed by the larger ML community. 
2. By changing the typical number of cells in each patch, we can easily obtain information about the cellular 
   environment at different resolution from local (few cells) and global (thousand of cells). 
   

## Installation
First, you need Python 3.8 and Pytorch (with CUDA support).
If you run the following command from your terminal it should report True:
```
python -c 'import torch; print(torch.cuda.is_available())'
```

Next install the most recent version of Pyro (not yet available using pip):
```bash
git clone https://github.com/pyro-ppl/pyro.git
cd pyro
pip install .
```

Finally install *Tissue Purifier* and its dependencies:
```bash
git clone https://github.com/broadinstitute/tissue_purifier.git
cd tissue_purifier
pip install -r requirements.txt
pip install .   
```

## Docker Image
A GPU-enabled docker image is available from the Google Container Registry (GCR) as:

``us.gcr.io/broad-dsde-methods/tissuepurifier:latest``

Older versions are available at the same location, for example as

``us.gcr.io/broad-dsde-methods/tissuepurifier:0.0.4``


### How to run
You can run the notebooks sequentially:
- [notebook1](https://github.com/broadinstitute/tissue_purifier/blob/main/notebooks/notebook1.ipynb).
- [notebook2](https://github.com/broadinstitute/tissue_purifier/blob/main/notebooks/notebook2.ipynb>).
- [notebook3](https://github.com/broadinstitute/tissue_purifier/blob/main/notebooks/notebook3.ipynb>).

or from the command line:
```bash
python main.py --from_yaml config.from_yaml  # train ssl
python analyze.py --anndata_in XXX --ckpt_file XXX.pt  # DOUBLE CHECK
python gene_regression.py --anndata_in XXX --l1  --l2 # DOUBLE CHECK
```

## Features and Limitations 
Features:
1. We have implemented multiple ssl strategies (such as convolutional Vae, Dino, BarlowTwin, SimClr) 
   based on recent advances in image-based Machine Learning. 
2. Tissue Purifier can be used to analyze any type of localized quantitative measurement for example spatial proteomics. (not only mRNA count data).

Limitations:
1. Currently, tissue purifier works only with 2D tissue slices
2. Currently, we assume a deterministic cell-type assignment

## Future Improvements
Future improvements:
1. 3D images
2. probabilistic cell-type assignment
3. pairing with histopathology (i.e. dense-image) 

### Citation   
This software package was developed by [Luca D'Alessio](dalessioluca@gmail.com) and 
[Fedor Grab](grab.f@northeastern.edu ) 

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
