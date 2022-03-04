__version__ = "0.0.1"

# The actual API
from tissue_purifier import data_utils as data
from tissue_purifier import gene_regression as genex
from tissue_purifier import io_utils as io
from tissue_purifier import model_utils as models
from tissue_purifier import plot_utils as plot
from tissue_purifier import misc_utils as utils


__all__ = [
    "data",
    "genex",
    "utils",
    "io",
    "models",
    "plot",
]

# it means that you can do
# import tissue_purifier as tp
# a = tp.data.AnndatFolderDM

