import torch
import numpy
from scanpy import AnnData
from typing import NamedTuple, Union, List, Any, Optional
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from tissue_purifier.utils.validation_util import SmartPca


def _make_labels(y: Union[torch.Tensor, numpy.ndarray, List[Any]]) -> (torch.Tensor, dict):
    def _to_numpy(_y):
        if isinstance(_y, numpy.ndarray):
            return _y
        elif isinstance(_y, list):
            return numpy.array(_y)
        elif isinstance(_y, torch.Tensor):
            return _y.detach().cpu().numpy()

    y_np = _to_numpy(y)
    unique_labels = numpy.unique(y_np)
    y_to_ids = dict(zip(unique_labels, numpy.arange(y_np.shape[0])))
    labels = torch.tensor([y_to_ids[tmp] for tmp in y_np])
    return labels, y_to_ids


class GeneDataset(NamedTuple):
    """
    Container for organizing the gene expression data
    """
    # order is important. Do not change

    #: float tensor with the covariates of shape (n, k)
    covariates: torch.Tensor

    #: long tensor with the cell_type_ids of shape (n)
    cell_type_ids: torch.Tensor

    #: long tensor with the count data of shape (n, g)
    counts: torch.Tensor

    #: number of cell types
    k_cell_types: int

    #: dictionary with mapping from unique_cell_type to cell_type_ids
    cell_type_mapping: dict

    #: list of the gene names
    gene_names: List[str]

    def describe(self):
        """ Method which described the content and the GeneDataset. """
        for k, v in zip(self._fields, self):
            if isinstance(v, int) or isinstance(v, dict):
                print("{} ---> {}".format(k.ljust(20), v))
            elif isinstance(v, list):
                print("{} ---> list of length {}".format(k.ljust(20), len(v)))
            else:
                print("{} ---> {}".format(k.ljust(20), v.shape))


def make_gene_dataset_from_anndata(
        anndata: AnnData,
        cell_type_key: str,
        covariate_key: str,
        preprocess_strategy: str = 'raw',
        apply_pca: bool = False,
        n_components: Union[int, float] = 0.9) -> GeneDataset:
    """
    Convert a anndata object into a GeneDataset object which can be used for gene regression.

    Args:
        anndata: AnnData object with the raw counts stored in anndata.X
        cell_type_key: key corresponding to the cell type, i.e. cell_types = anndata.obs[cell_type_key]
        covariate_key: key corresponding to the covariate, i.e. covariates = anndata.obsm[covariate_key]
        preprocess_strategy: either 'center', 'z_score' or 'raw'. It describes how to preprocess the covariates.
            'raw' (default) means no preprocessing.
        apply_pca: if True, we compute the pca of the covariates. This operation happens after the preprocessing.
        n_components: Used only if :attr:`apply_pca` == True.
            If integer specifies the dimensionality of the data after PCA.
            If float in (0, 1) it auto selects the dimensionality so that the explained variance is at least that value.

    Returns:
        GeneDataset: a GeneDataset object
    """

    def _to_torch(_x):
        if isinstance(_x, torch.Tensor):
            _y = _x.detach().cpu()
        elif isinstance(_x, numpy.ndarray):
            _y = torch.from_numpy(_x)
        else:
            raise Exception("Error. Expected torch.Tensor or numpy.array. Received {}".format(type(_x)))
        assert torch.all(torch.isfinite(_y)), "Error. Tensor is not finite {}".format(_y)
        return _y

    assert preprocess_strategy in {'center', 'z_score', 'raw'}, \
        'Preprocess strategy must be either "center", "z_score" or "raw"'

    assert (isinstance(n_components, int) and n_components >= 1) or \
        (isinstance(n_components, float) and 0.0 < n_components < 1.0), \
        "n_components must be a positive integer or a float in (0.0, 1.0)"

    assert cell_type_key in set(anndata.obs.keys()), "Cell_type_key is not present in anndata.obs.keys()"
    assert covariate_key in set(anndata.obsm.keys()), "Covariate_key is not present in anndata.obsm.keys()"

    cell_types = list(anndata.obs[cell_type_key].values)
    cell_type_ids_n, mapping_dict = _make_labels(cell_types)
    counts_ng = torch.tensor(anndata.X.toarray()).long()
    covariates_nl_raw = torch.tensor(anndata.obsm[covariate_key])

    if not torch.all(torch.isfinite(covariates_nl_raw)):
        mask = torch.isfinite(covariates_nl_raw)
        raise ValueError("covariates in the anndata file are not finite {}.".format(covariates_nl_raw[~mask]))

    assert len(counts_ng.shape) == 2
    assert len(covariates_nl_raw.shape) == 2
    assert len(cell_type_ids_n.shape) == 1

    assert counts_ng.shape[0] == covariates_nl_raw.shape[0] == cell_type_ids_n.shape[0]

    k_cell_types = cell_type_ids_n.max().item() + 1  # +1 b/c ids start from zero

    if apply_pca:
        new_covariate = SmartPca(preprocess_strategy=preprocess_strategy).fit_transform(data=covariates_nl_raw,
                                                                                        n_components=n_components)
    elif preprocess_strategy == 'z_score':
        std, mean = torch.std_mean(covariates_nl_raw, dim=-2, unbiased=True, keepdim=True)
        mask = (std == 0.0)
        std[mask] = 1.0
        new_covariate = (covariates_nl_raw - mean) / std
    elif preprocess_strategy == 'center':
        mean = torch.mean(covariates_nl_raw, dim=-2, keepdim=True)
        new_covariate = covariates_nl_raw - mean
    else:
        new_covariate = covariates_nl_raw

    return GeneDataset(
        cell_type_ids=_to_torch(cell_type_ids_n),
        covariates=_to_torch(new_covariate),
        counts=_to_torch(counts_ng),
        k_cell_types=k_cell_types,
        cell_type_mapping=mapping_dict,
        gene_names=list(anndata.var_names),
    )


def train_test_val_split(
        data: Union[List[torch.Tensor], List[numpy.ndarray], GeneDataset],
        train_size: float = 0.8,
        test_size: float = 0.15,
        val_size: float = 0.05,
        n_splits: int = 1,
        random_state: int = None,
        stratify: bool = True):
    """
    Utility function used to split the data into train/test/val.

    Args:
        data: the data to split into train/test/val
        train_size: the relative size of the train dataset
        test_size: the relative size of the test dataset
        val_size: the relative size of the val dataset
        n_splits: how many times to split the data
        random_state: specify the random state for reproducibility
        stratify: If true the tran/test/val are stratified so that they contain approximately the same
            number of example from each class. If data is a list of arrays the 2nd array is assumed to represent the
            class. If data is a GeneDataset the class is the cell_type.

    Returns:
        tuple: yields multiple splits of the data.

    Example:
          >>> for train, test, val in train_test_val_split(data=[x,y,z]):
          >>>       x_train, y_train, z_train = train
          >>>       x_test, y_test, z_test = test
          >>>       x_val, y_val, z_val = val
          >>>       ... do something ...

    Example:
          >>> for train, test, val in train_test_val_split(data=GeneDataset):
          >>>       assert isinstance(train, GeneDataset)
          >>>       assert isinstance(test, GeneDataset)
          >>>       assert isinstance(val, GeneDataset)
          >>>       ... do something ...
    """

    if train_size <= 0:
        raise ValueError("Train_size must be > 0")
    if test_size <= 0:
        raise ValueError("Test_size must be > 0")
    if val_size <= 0:
        raise ValueError("Val_size must be > 0")

    if isinstance(data, List):
        arrays = data
    elif isinstance(data, GeneDataset):
        # same order as in the definition of GeneDataset NamedTuple
        arrays = [data.covariates, data.cell_type_ids, data.counts]
    else:
        raise ValueError("data must be a list or a GeneDataset")

    dims_actual = [a.shape[0] for a in arrays]
    dims_expected = [dims_actual[0]] * len(dims_actual)
    assert all(a == b for a, b in zip(dims_actual, dims_expected)), \
        "Error. All leading dimensions should be the same"

    # Normalize the train/test/val sizes
    norm0 = train_size + test_size + val_size
    train_size_norm0 = train_size / norm0
    test_and_val_size_norm0 = (test_size + val_size) / norm0

    norm1 = test_size + val_size
    test_size_norm1 = test_size / norm1
    val_size_norm1 = val_size / norm1

    if stratify:
        sss0 = StratifiedShuffleSplit(n_splits=n_splits,
                                      train_size=train_size_norm0,
                                      test_size=test_and_val_size_norm0,
                                      random_state=random_state)
        sss1 = StratifiedShuffleSplit(n_splits=1,
                                      train_size=val_size_norm1,
                                      test_size=test_size_norm1,
                                      random_state=random_state)
    else:
        sss0 = ShuffleSplit(n_splits=n_splits,
                            train_size=train_size_norm0,
                            test_size=test_and_val_size_norm0,
                            random_state=random_state)
        sss1 = ShuffleSplit(n_splits=1,
                            train_size=val_size_norm1,
                            test_size=test_size_norm1,
                            random_state=random_state)

    # Part common to both stratified and not stratified
    trains, tests, vals = None, None, None
    for index_train, index_test_and_val in sss0.split(*arrays[:2]):
        trains = [a[index_train] for a in arrays]
        test_and_val = [a[index_test_and_val] for a in arrays]
        for index_val, index_test in sss1.split(*test_and_val[:2]):
            tests = [a[index_test] for a in test_and_val]
            vals = [a[index_val] for a in test_and_val]

        if isinstance(data, GeneDataset):
            # copy the k_cell_types and cell_type_mapping in the train/test/val
            trains += [data.k_cell_types, data.cell_type_mapping, data.gene_names]
            tests += [data.k_cell_types, data.cell_type_mapping, data.gene_names]
            vals += [data.k_cell_types, data.cell_type_mapping, data.gene_names]
            yield GeneDataset._make(trains), GeneDataset._make(tests), GeneDataset._make(vals)
        else:
            yield trains, tests, vals
