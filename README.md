[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Tissue Purifier 
Tissue Purifier (TP) is a tool to perform tissue analysis in python.
Tissue are cropped into overlapping patches, semantic features are associated to each patch via self-supervised (contrastive) 
learning. The learned features are used in downstram tasks (such as differential gene expression analysis).

## Installation
create a virtual environment and install pytorch then clone the repo, navigate to root folder, install with pip
> pip install -r requirements.txt
> git clone https://github.com/broadinstitute/tissue_purifier.git \
> cd tissue_purifier \
> pip install -e .[tests] 


