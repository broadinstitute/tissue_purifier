import torch
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.base import BaseEstimator
from tissue_purifier import data_utils
from tissue_purifier.data_utils.transforms import Rasterize

# TODO: This file need to be completely rewritten


def get_dense_image(df, pixel_size):
    t = Rasterize(sigma=(1.0, 1.0))(
        data_utils.SparseImage.from_panda(
            df, x="x", y="y", category="cell_type", pixel_size=pixel_size, padding=10
        ).to_dense()
    )
    
    arr = np.zeros(t.shape)
    for i in range(9):
        arr[i] = minmax_scale(t[i].cpu())
        
    return arr.transpose((1, 2, 0))


def get_sickness_mask(image, labels_or_probas, labels, current_image_label, config, threshold=0.5, return_probas=False):
    intensity_mask = np.zeros(image.shape[:-1])
    count_mask = np.zeros(image.shape[:-1])
    offset = 0
    crop_size = int(config["simulation"]["CROP_SIZE"])
    raise NotImplementedError
    xys = cache["test"]
    
    for i, (proba, label) in enumerate(zip(labels_or_probas, labels)):
        if label == current_image_label:
            crop_i, crop_j = xys[i]
            if type(proba) == torch.Tensor:
                proba = proba.cpu().detach().numpy()
                
            intensity_mask[
                crop_i + offset: crop_i + crop_size - offset, crop_j + offset: crop_j + crop_size - offset
            ] += proba
            count_mask[
                crop_i + offset: crop_i + crop_size - offset,
                crop_j + offset: crop_j + crop_size - offset
            ] += 1
       
    proba_mask = intensity_mask / count_mask
    proba_mask = np.where(proba_mask == np.inf, 0, proba_mask)
    proba_mask = np.where(proba_mask == np.nan, 0, proba_mask)
    
    if return_probas:
        return proba_mask

    return np.where(proba_mask > threshold, 1, 0)


def create_mask_for_sample(df, encoder, cluster_alg, config, proba=False, threshold=0.5, n_crops=3000):
    sparse_images = data_utils.SparseImage.from_panda(
        df, x="x", y="y", category="cell_type", pixel_size=config["simulation"]["PIXEL_SIZE"], padding=10
    )
    testloader = data_utils.helpers.define_testloader(
        sparse_images, [0], [0], config, n_crops_for_tissue=n_crops
    )
    is_sklearn_alg = isinstance(cluster_alg, BaseEstimator)
    embeddings, labels, fnames = encoder.embed(testloader, to_numpy=is_sklearn_alg)
    
    img = get_dense_image(df, config["simulation"]["PIXEL_SIZE"])
    
    if is_sklearn_alg:
        clusters = cluster_alg.fit_predict(embeddings)
    else:
        clusters = torch.softmax(cluster_alg(embeddings), dim=-1)[:, 1]
    return get_sickness_mask(img, clusters, labels, 0, config, return_probas=proba, threshold=threshold)


def _get_probability_for_bead(row, mask, x_min, y_min, pixel_size):
    i = int((row.y - y_min) / pixel_size)
    j = int((row.x - x_min) / pixel_size)
    return mask[j, i]


def spatialize_mask(df, mask, pixel_size):
    x_min, y_min = df.x.min(), df.y.min()
    return df.apply(
        lambda row: _get_probability_for_bead(row, mask=mask, x_min=x_min, y_min=y_min, pixel_size=pixel_size),
        axis=1
    )
