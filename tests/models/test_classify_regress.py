import pytest
import torch
import numpy
import pandas
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tissue_purifier.model_utils.classify_regress import (
    classify_and_regress,
    PlRegressor,
    PlClassifier)


@pytest.mark.parametrize("x_type", ["torch", "numpy"])
@pytest.mark.parametrize("y_type", ["torch", "numpy", "list"])
def test_regressor(x_type, y_type):
    max_iter = 5
    n, d = 10, 3
    X = torch.randn(n, d)
    y = torch.randn(n)
    if x_type == "numpy":
        X = X.cpu().numpy()
    if y_type == "numpy":
        y = y.cpu().numpy()
    elif y_type == "list":
        y = y.cpu().tolist()

    regressor = PlRegressor(max_iter=max_iter)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    score = regressor.score(X, y)
    assert isinstance(score, float)
    assert isinstance(y_pred, numpy.ndarray)
    assert y_pred.shape[0] == n
    loss = regressor.loss_
    loss_curve = regressor.loss_curve_
    assert isinstance(loss, float)
    assert isinstance(loss_curve, list) and len(loss_curve) == max_iter


@pytest.mark.parametrize("x_type", ["torch", "numpy"])
@pytest.mark.parametrize("y_type", ["torch", "numpy", "list"])
def test_classifier(x_type, y_type):
    max_iter = 5
    n, d = 10, 3
    X = torch.randn(n, d)
    y = torch.randint(low=3, high=5, size=[n])
    if x_type == "numpy":
        X = X.cpu().numpy()
    if y_type == "numpy":
        y = y.cpu().numpy()
    elif y_type == "list":
        y = y.cpu().tolist()

    classifier = PlClassifier(max_iter=max_iter)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    score = classifier.score(X, y)
    prob = classifier.predict_proba(X)
    log_prob = classifier.predict_log_proba(X)
    assert isinstance(score, float)
    assert isinstance(y_pred, numpy.ndarray)
    assert isinstance(prob, numpy.ndarray)
    assert isinstance(log_prob, numpy.ndarray)
    assert prob.shape == log_prob.shape
    assert y_pred.shape[0] == n

    loss = classifier.loss_
    loss_curve = classifier.loss_curve_
    assert isinstance(loss, float)
    assert isinstance(loss_curve, list) and len(loss_curve) == max_iter


@pytest.mark.parametrize("hard_bootstrapping", [True, False])
@pytest.mark.parametrize("x_type", ["torch", "numpy"])
@pytest.mark.parametrize("y_type", ["torch", "numpy", "list"])
def test_noisy_classifier(x_type, y_type, hard_bootstrapping):
    max_iter = 5
    n, d = 10, 3
    X = torch.randn(n, d)
    y = torch.randint(low=3, high=5, size=[n])
    if x_type == "numpy":
        X = X.cpu().numpy()
    if y_type == "numpy":
        y = y.cpu().numpy()
    elif y_type == "list":
        y = y.cpu().tolist()

    classifier = PlClassifier(
        max_iter=max_iter,
        noisy_labels=True,
        bootstrap_epoch_start=3,
        lambda_reg=1.0,
        hard_bootstrapping=hard_bootstrapping)

    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    score = classifier.score(X, y)
    prob = classifier.predict_proba(X)
    log_prob = classifier.predict_log_proba(X)
    assert isinstance(score, float)
    assert isinstance(y_pred, numpy.ndarray)
    assert isinstance(prob, numpy.ndarray)
    assert isinstance(log_prob, numpy.ndarray)
    assert prob.shape == log_prob.shape
    assert y_pred.shape[0] == n

    loss = classifier.loss_
    loss_curve = classifier.loss_curve_
    assert isinstance(loss, float)
    assert isinstance(loss_curve, list) and len(loss_curve) == max_iter


@pytest.mark.parametrize("n_splits, n_repeats", [(1, 1), (2, 1), (2, 2)])
def test_classify_regress_knn(n_splits, n_repeats, capsys):
    """ Test that knn classifier regressor works """

    n, p = 23, 12
    my_dict = {
        "feature_a": torch.randn((n, p)),
        "feature_b": torch.randn((n, 2*p)),
        "label_a": torch.randint(low=0, high=5, size=[n]),
        "label_b": ["wt" if i < 0.5*n else "dis" for i in range(n)],
        "label_c": torch.zeros(n),
        "regress_a": torch.randn(n),
        "regress_b": numpy.random.randn(n)
    }

    feature_keys, classify_keys, regress_keys = [], [], []
    for k in my_dict.keys():
        if k.startswith("feature"):
            feature_keys.append(k)
        elif k.startswith("label"):
            classify_keys.append(k)
        elif k.startswith("regress"):
            regress_keys.append(k)

    def exclude_self(d):
        w = numpy.ones_like(d)
        w[d == 0.0] = 0.0
        return w

    kn_kargs = {
        "n_neighbors": 5,
        "weights": exclude_self,
    }

    df_tot = classify_and_regress(
        input_dict=my_dict,
        feature_keys=feature_keys,
        regress_keys=regress_keys,
        classify_keys=classify_keys,
        regressor=KNeighborsRegressor(**kn_kargs),
        classifier=KNeighborsClassifier(**kn_kargs),
        n_splits=n_splits,
        n_repeats=n_repeats)

    df_tot["combined_key"] = df_tot["x_key"] + "_" + df_tot["y_key"]
    assert isinstance(df_tot, pandas.DataFrame)
    assert df_tot.shape[0] == n_repeats * n_splits * len(feature_keys) * (len(regress_keys) + len(classify_keys))

#    with capsys.disabled():
#        # inside this context the stdout will not be captured
#        print(n_repeats, n_splits)
#        df_mean = df_tot.groupby("combined_key").mean()
#        for row in df_mean.itertuples():
#            for k, v in row._asdict().items():
#                if isinstance(v, float) and numpy.isfinite(v):
#                    name = "kn/" + row.Index + "/" + k + "/mean"
#                    print(name, v)
#
#        df_std = df_tot.groupby("combined_key").std()
#        for row in df_std.itertuples():
#            for k, v in row._asdict().items():
#                if isinstance(v, float) and numpy.isfinite(v):
#                    name = "kn/" + row.Index + "/" + k + "/std"
#                    print(name, v)
