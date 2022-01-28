import numpy
import torch
from neptune.new.types import File
from pytorch_lightning import LightningModule
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import RidgeClassifierCV, RidgeCV


from tissue_purifier.plot_utils.plot_embeddings import plot_embeddings
from tissue_purifier.model_utils.classify_regress import classify_and_regress
from tissue_purifier.misc_utils.nms import NonMaxSuppression
from tissue_purifier.misc_utils.dict_util import (
    concatenate_list_of_dict,
    subset_dict)

from tissue_purifier.misc_utils.misc import (
    SmartPca,
    SmartUmap)


def knn_classification(world_dict: dict, val_iomin_threshold: float):
    """
    Make subdictionary with non-overlapping patches.
    """
    assert {"patches_xywh", "classify_tissue_label"}.issubset(world_dict.keys())

    # Classification
    feature_keys, regress_keys, classify_keys = [], [], []
    for key in world_dict.keys():
        if key.startswith("regress"):
            regress_keys.append(key)
        elif key.startswith("classify"):
            classify_keys.append(key)
        elif key.startswith("pca_") or key.startswith("umap_") or key.startswith("feature"):
            feature_keys.append(key)

    # KNN
    def exclude_self(d):
        w = numpy.ones_like(d)
        w[d == 0.0] = 0.0
        return w

    kn_kargs = {
        "n_neighbors": 5,
        "weights": exclude_self,
    }

    regressor = KNeighborsRegressor(**kn_kargs)
    classifier = KNeighborsClassifier(**kn_kargs)

    # loop over subset made of non-overlapping patches
    df_tot = None

    # compute the patch_to_patch overlap just one at the beginning
    patches = world_dict["patches_xywh"]
    initial_score = torch.rand_like(patches[:, 0].float())
    tissue_ids = world_dict["classify_tissue_label"]
    nms_mask_n, overlap_nn = NonMaxSuppression.compute_nm_mask(
        score=initial_score,
        ids=tissue_ids,
        patches_xywh=patches,
        iom_threshold=val_iomin_threshold)
    binarized_overlap_nn = (overlap_nn > val_iomin_threshold).float()

    for n in range(20):
        # create a dictionary with only non-overlapping patches to test kn-regressor/classifier
        nms_mask_n = NonMaxSuppression._perform_nms_selection(mask_overlap_nn=binarized_overlap_nn,
                                                              score_n=torch.rand_like(initial_score),
                                                              possible_n=torch.ones_like(initial_score).bool())
        world_dict_subset = subset_dict(input_dict=world_dict, mask=nms_mask_n)

        df_tmp = classify_and_regress(
            input_dict=world_dict_subset,
            feature_keys=feature_keys,
            regress_keys=regress_keys,
            classify_keys=classify_keys,
            regressor=regressor,
            classifier=classifier,
            n_repeats=1,
            n_splits=1,
            verbose=False,
        )
        df_tot = df_tmp if df_tot is None else df_tot.merge(df_tmp, how='outer')

    df_tot["combined_key"] = df_tot["x_key"] + "_" + df_tot["y_key"]
    df_mean = df_tot.groupby("combined_key").mean()
    df_std = df_tot.groupby("combined_key").std()
    return df_mean, df_std


def linear_classification(world_dict: dict):
    # Classification/regression
    feature_keys, regress_keys, classify_keys = [], [], []
    for key in world_dict.keys():
        if key.startswith("regress"):
            regress_keys.append(key)
        elif key.startswith("classify"):
            classify_keys.append(key)
        elif key.startswith("pca_") or key.startswith("umap_") or key.startswith("feature"):
            feature_keys.append(key)

    ridge_kargs = {
        "alphas": (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    }

    regressor = RidgeCV(**ridge_kargs)
    classifier = RidgeClassifierCV(**ridge_kargs)

    # loop over subset made of non-overlapping patches
    df_tot = classify_and_regress(
        input_dict=world_dict,
        feature_keys=feature_keys,
        regress_keys=regress_keys,
        classify_keys=classify_keys,
        regressor=regressor,
        classifier=classifier,
        n_repeats=5,
        n_splits=5,
        verbose=False,
    )

    df_tot["combined_key"] = df_tot["x_key"] + "_" + df_tot["y_key"]
    df_mean = df_tot.groupby("combined_key").mean()
    df_std = df_tot.groupby("combined_key").std()
    return df_mean, df_std


class BenchmarkModel(LightningModule):
    """
    Model with the routine to evaluate the embeddings
    """
    def __init__(self, val_iomin_threshold: float):
        super(BenchmarkModel, self).__init__()
        self.val_iomin_threshold = val_iomin_threshold

    def validation_epoch_end(self, list_of_val_dict) -> None:
        """ You can receive a list_of_valdict or, if you have multiple val_datasets a list_of_list_of_valdict """
        print("inside validation epoch end")

        if isinstance(list_of_val_dict[0], dict):
            list_dict = [concatenate_list_of_dict(list_of_val_dict)]
        elif isinstance(list_of_val_dict[0], list):
            list_dict = [concatenate_list_of_dict(tmp_list) for tmp_list in list_of_val_dict]
        else:
            raise Exception("In validation epoch end. I received an unexpected input")

        for loader_idx, total_dict in enumerate(list_dict):
            print("rank {0} dataloader_idx {1}".format(self.global_rank, loader_idx))

            # gather dictionaries from the other processes and flatten the extra dimension.
            world_dict = self.all_gather(total_dict)
            all_keys = list(world_dict.keys())
            for key in all_keys:
                if len(world_dict[key].shape) == 1 + len(total_dict[key].shape):
                    world_dict[key] = world_dict[key].flatten(end_dim=1)
            print("done dictionary. rank {0}".format(self.global_rank))

            # DO operations ONLY on rank 0.
            # ADD "rank_zero_only=True" to avoid deadlocks on synchronization.
            if self.global_rank == 0:

                # plot the UMAP colored by all available annotations
                smart_pca = SmartPca(preprocess_strategy='z_score')
                smart_umap = SmartUmap(n_neighbors=25, preprocess_strategy='raw',
                                       n_components=2, min_dist=0.5, metric='euclidean')

                embedding_keys = []
                annotation_keys = []
                all_keys = list(world_dict.keys())
                for k in all_keys:
                    if k.startswith("feature"):
                        embedding_keys.append(k)
                        # print("working on feature", k)
                        input_features = world_dict[k]
                        embeddings_pca = smart_pca.fit_transform(input_features, n_components=0.95)
                        embeddings_umap = smart_umap.fit_transform(embeddings_pca)
                        world_dict['pca_' + k] = embeddings_pca
                        world_dict['umap_' + k] = embeddings_umap
                    elif k.startswith("regress") or k.startswith("classify"):
                        annotation_keys.append(k)

                all_files = []
                for embedding_key in embedding_keys:
                    fig_tmp = plot_embeddings(
                        input_dictionary=world_dict,
                        embedding_key=embedding_key,
                        annotation_keys=annotation_keys,
                        n_col=2,
                    )
                    all_files.append(File.as_image(fig_tmp))

                for file_tmp, key_tmp in zip(all_files, embedding_keys):
                    self.logger.run["maps/" + key_tmp].log(file_tmp)
                # print("printed the embeddings")

                # knn classification/regression
                df_mean_knn, df_std_knn = knn_classification(world_dict, self.val_iomin_threshold)
                # print("df_mean_knn ->", df_mean_knn)

                for row in df_mean_knn.itertuples():
                    for k, v in row._asdict().items():
                        if isinstance(v, float) and numpy.isfinite(v):
                            name = "kn/" + row.Index + "/" + k + "/mean"
                            self.log(name=name, value=v, batch_size=1, rank_zero_only=True)

                for row in df_std_knn.itertuples():
                    for k, v in row._asdict().items():
                        if isinstance(v, float) and numpy.isfinite(v):
                            name = "kn/" + row.Index + "/" + k + "/std"
                            self.log(name=name, value=v, batch_size=1, rank_zero_only=True)

                # linear classification/regression
                df_mean_linear, df_std_linear = linear_classification(world_dict)
                # print("df_mean_linear ->", df_mean_linear)

                for row in df_mean_linear.itertuples():
                    for k, v in row._asdict().items():
                        if isinstance(v, float) and numpy.isfinite(v):
                            name = "linear/" + row.Index + "/" + k + "/mean"
                            self.log(name=name, value=v, batch_size=1, rank_zero_only=True)

                for row in df_std_linear.itertuples():
                    for k, v in row._asdict().items():
                        if isinstance(v, float) and numpy.isfinite(v):
                            name = "linear/" + row.Index + "/" + k + "/std"
                            self.log(name=name, value=v, batch_size=1, rank_zero_only=True)
