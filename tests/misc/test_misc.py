import torch
from tissue_purifier.misc_utils.misc import get_percentile


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
