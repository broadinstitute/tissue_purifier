import torch
from tissue_purifier.misc_utils.misc import get_percentile, SmartPca
import pytest
from sklearn.decomposition import PCA
import numpy

def test_get_percentile(capsys):
    """ Transform a tensor into its percentiles which are in [0.0, 1.0] """
    data = torch.zeros((2, 10))
    data[0] = torch.arange(10)
    data[1] = -torch.arange(10)
    q_true = torch.zeros_like(data)
    q_true[0] = torch.linspace(0.0, 1.0, 10)
    q_true[1] = torch.linspace(1.0, 0.0, 10)

    q = get_percentile(data, dim=-1)
    diff = (q - q_true).abs()
    assert torch.all(diff < 1E-4)

    data = torch.randn((2, 10))
    q_true = torch.zeros_like(data)
    mask_data0_le_data1 = data[0] <= data[1]
    q_true[0, mask_data0_le_data1] = 0.0
    q_true[1, mask_data0_le_data1] = 1.0
    q_true[0, ~mask_data0_le_data1] = 1.0
    q_true[1, ~mask_data0_le_data1] = 0.0
    q = get_percentile(data, dim=-2)
    diff = (q - q_true).abs()
    assert torch.all(diff < 1E-4)


@pytest.mark.parametrize("n, p", [[200, 30], [20, 300]])
def test_pca(n, p):
    q = 0.75
    data = torch.randn((n, p))
    pca = SmartPca(preprocess_strategy='raw')
    x1 = pca.fit_transform(data, n_components=q)
    nq = x1.shape[-1]
    assert (pca.explained_variance_ratio_[nq] > q)


@pytest.mark.parametrize("n, q, p", [[20, 2, 200], [20, 2, 3000]])
def test_pca_vs_scikit(n, q, p, capsys):
    """ Compare myPCA vs scikit-learn PCA"""
    # n = sample
    # q = low dimension (a bit larger than true rank), i.e. output of PCA
    # p = features
    k = 2  # true rank

    if torch.cuda.is_available():
        U_true = torch.randn((n, k), device=torch.device('cuda'))
        S_true = torch.randn(k, device=torch.device('cuda')).exp()
        V_true = torch.randn((k, p), device=torch.device('cuda'))
    else:
        U_true = torch.randn((n, k), device=torch.device('cpu'))
        S_true = torch.randn(k, device=torch.device('cpu')).exp()
        V_true = torch.randn((k, p), device=torch.device('cpu'))

    color = torch.zeros(n)
    color[:int(0.25 * n)] = 0
    color[int(0.25 * n):int(0.5 * n)] = 1
    color[int(0.5 * n):int(0.75 * n)] = 2
    color[int(0.75 * n):] = 3

    dx, dy = 3, 5
    U_true[color == 0] += torch.tensor([dx, dy])
    U_true[color == 1] += torch.tensor([dx, -dy])
    U_true[color == 2] += torch.tensor([-dx, dy])
    U_true[color == 3] += torch.tensor([-dx, -dy])
    # plt.scatter(U_true[:, 0], U_true[:, 1], c=color)

    data = U_true @ torch.diag(S_true) @ V_true
    pca = SmartPca(preprocess_strategy='raw')
    x1 = pca.fit_transform(data, n_components=q)

    # Compare against PCA in scikit-learn
    pca_scikit = PCA(n_components=q, copy=False, whiten=False,
                     svd_solver='auto', tol=0.0,
                     iterated_power='auto', random_state=0)
    x2 = pca_scikit.fit_transform(data.detach().cpu().numpy())

    # check that the coordinates are correlated between myPCA and scikitPCA
    x1_mean = numpy.mean(x1, axis=-2)
    x2_mean = numpy.mean(x2, axis=-2)
    cov = numpy.mean((x1-x1_mean)*(x2-x2_mean), axis=-2)
    sigma1 = numpy.std(x1, ddof=1, axis=-2)
    sigma2 = numpy.std(x2, ddof=1, axis=-2)
    corr = cov / (sigma1 * sigma2)
    corr_abs = numpy.abs(corr)

    # check that the explained variance works
    ex_var1 = pca.explained_variance_[:q].detach().cpu().numpy()
    ex_var2 = pca_scikit.explained_variance_
    variance_error = numpy.abs(ex_var1-ex_var2) / ex_var2

    # with capsys.disabled():
    #     print("corr ->", corr)
    #     print("variance_error ->", variance_error)

    assert numpy.all(variance_error < 0.1)
    assert numpy.all(corr_abs > 0.9)
