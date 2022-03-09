Models
======

Mlp Classification and Regression
---------------------------------
We have implemented a Multi-Layer Perceptron (MLP) Classifier and Regressor with an interface similar
to the one in scikit-learn. Our implementation is much faster since it is based on pytorch and runs on GPUs.
Moreover, we have reimplemented the method described in the
`Unsupervised Label Noise Modeling and Loss Correction <https://arxiv.org/abs/1904.11238>`_ for classification
with noisy labels.

.. autoclass:: tissue_purifier.models.classifier_regressor.scikit_learn_interface.BaseEstimator

.. autoclass:: tissue_purifier.models.classifier_regressor.scikit_learn_interface.MlpRegressor
   :show-inheritance:
   :members: fit, predict, score, is_classifier, is_regressor

.. autoclass:: tissue_purifier.models.classifier_regressor.scikit_learn_interface.MlpClassifier
   :show-inheritance:
   :members: fit, predict, score, predict_proba, predict_log_proba, is_classifier, is_regressor


Self Supervised Models
----------------------
We have implemented multiple self-supervised learning (ssl) models.
They all inherit from the base class :class:`tissue_purifier.models.ssl_models.SslModelBase` which
is responsible for the validation (which is common to all ssl models) and logging.

.. autoclass:: tissue_purifier.models.ssl_models._ssl_base_model.SslModelBase

.. autoclass:: tissue_purifier.models.ssl_models.barlow.BarlowModel
   :show-inheritance:
   :members: add_specific_args, get_default_params

.. autoclass:: tissue_purifier.models.ssl_models.dino.DinoModel
   :show-inheritance:
   :members: add_specific_args, get_default_params

.. autoclass:: tissue_purifier.models.ssl_models.simclr.SimclrModel
   :show-inheritance:
   :members: add_specific_args, get_default_params

.. autoclass:: tissue_purifier.models.ssl_models.vae.VaeModel
   :show-inheritance:
   :members: add_specific_args, get_default_params

