import logging
from typing import Optional

import hdbscan
import numpy as np
import sklearn
import umap
from mteb.evaluation.evaluators.Evaluator import Evaluator
from river import cluster, stream
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)


class DBSTREAMWrapper(cluster.DBSTREAM):
    """Wrapper for river.cluster.DBSTREAM so it can be used similarly to sklearn API. Wrapper is used to store
    online predictions in self.labels_ attribute (as in sklearn), i.e., any new predictions will be added to this attribute.
    """

    def __init__(self, compute_labels: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_labels = compute_labels
        self.labels_ = []

    def _learn_many(self, X, sample_weights: Optional[list] = None):
        """Update model with multiple data points."""
        sample_weights = [None] * len(X) if sample_weights is None else sample_weights
        for i, (x, _) in enumerate(stream.iter_array(X)):
            self.learn_one(x, sample_weight=sample_weights[i])

    def _predict_many(self, X: np.array, sample_weights: Optional[list] = None) -> list:
        """Predict multiple data points."""
        labels = []
        sample_weights = [None] * len(X) if sample_weights is None else sample_weights
        for i, (x, _) in enumerate(stream.iter_array(X)):
            labels.append(self.predict_one(x, sample_weight=sample_weights[i]))
        return labels

    def fit(self, X: np.array, sample_weights: Optional[list] = None):
        """sklearn API logic to handle model updates."""
        self._learn_many(X, sample_weights=sample_weights)
        if self.compute_labels:
            self.predict(X, sample_weights=sample_weights)
        return self

    def fit_predict(self, X: np.array, sample_weights: Optional[list] = None):
        """sklearn API logic to handle simultaneous model updates and predictions."""
        self._learn_many(X, sample_weights=sample_weights)
        return self.predict(X, sample_weights=sample_weights)

    def predict(self, X: np.array, sample_weights: Optional[list] = None):
        """sklearn API logic to handle model predictions."""
        self.labels_.extend(self._predict_many(X, sample_weights=sample_weights))
        return self.labels_


class FlexibleClusteringEvaluator(Evaluator):
    def __init__(
        self,
        sentences: list[str],
        labels: list[int],
        clustering_alg: str = "minibatch_kmeans",
        clustering_params: Optional[dict] = None,
        dim_red: Optional[str] = None,
        dim_red_params: Optional[dict] = None,
        limit: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        self.sentences = sentences
        self.labels = labels

        if clustering_params is None:
            clustering_params = {}

        if dim_red_params is None:
            dim_red_params = {}

        nr_labels = len(set(self.labels))
        if clustering_alg == "agglomerative":
            self.clustering_model = sklearn.cluster.AgglomerativeClustering(
                n_clusters=nr_labels, **clustering_params
            )
        elif clustering_alg == "dbstream":
            self.clustering_model = DBSTREAMWrapper(**clustering_params)

        elif clustering_alg == "hdbscan":
            self.clustering_model = hdbscan.HDBSCAN(**clustering_params)

        elif clustering_alg == "minibatch_kmeans":
            if "batch_size" not in clustering_params:
                clustering_params["batch_size"] = 500
            if "n_init" not in clustering_params:
                clustering_params["n_init"] = "auto"
            self.clustering_model = sklearn.cluster.MiniBatchKMeans(
                n_clusters=nr_labels, **clustering_params
            )
        else:
            raise ValueError("Option not implemented")

        if dim_red == "pca":
            self.dim_red_model = sklearn.decomposition.PCA(
                n_components=2, **dim_red_params
            )
        elif dim_red == "umap":
            self.dim_red_model = umap.UMAP(metric="cosine", **dim_red_params)
        elif dim_red == "pca+umap":
            self.dim_red_model = make_pipeline(
                sklearn.decomposition.PCA(n_components=50),
                umap.UMAP(metric="cosine", **dim_red_params),
            )
        elif dim_red is None:
            self.dim_red_model = None
        else:
            raise ValueError("Option not implemented")

        self.model_name = f"{dim_red}>{clustering_alg}" if dim_red else clustering_alg

    def __call__(self, model):
        logger.info(f"Encoding {len(self.sentences)} sentences...")
        corpus_embeddings = np.asarray(model.encode(self.sentences))

        logger.info(f"Fitting {self.model_name} model...")
        if self.dim_red_model is not None:
            corpus_embeddings = self.dim_red_model.fit_transform(corpus_embeddings)
        self.clustering_model.fit(corpus_embeddings)
        cluster_assignment = self.clustering_model.labels_

        logger.info("Evaluating...")
        v_measure = sklearn.metrics.cluster.v_measure_score(
            self.labels, cluster_assignment
        )

        return {"v_measure": v_measure}
