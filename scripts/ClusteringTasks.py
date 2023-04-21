import json
import os
from pathlib import Path
from typing import Callable, Optional, TypeVar

import datasets
import numpy as np
import tqdm
from FlexibleClusteringEvaluator import FlexibleClusteringEvaluator
from mteb.abstasks.AbsTask import AbsTask
from sentence_transformers import SentenceTransformer
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def dynamic_description(d: Callable[P, T]) -> Callable[P, T]:
    @property
    def wrapper(self) -> dict:
        desc = d(self)
        desc["name"] = (
            desc["name"]
            + "{"
            + (self.dim_red + "," if self.dim_red else "")
            + self.clustering_alg
            + (
                "{"
                + ",".join(
                    [
                        str(key) + "=" + str(value)
                        for key, value in self.clustering_params.items()
                    ]
                )
                + "}"
                if len(self.clustering_params) > 0
                else ""
            )
            + "}"
        )
        return desc

    return wrapper


class AbsTaskFlexibleClustering(AbsTask):
    def __init__(
        self,
        clustering_alg: str = "minibatch_kmeans",
        clustering_params: Optional[dict] = None,
        dim_red: Optional[str] = None,
        dim_red_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.clustering_alg = clustering_alg
        self.dim_red = dim_red

        if clustering_params is None:
            self.clustering_params = {}
        else:
            self.clustering_params = clustering_params
        if dim_red_params is None:
            self.dim_red_params = {}
        else:
            self.dim_red_params = dim_red_params

    def evaluate(self, model, split: str = "test", **kwargs):
        if not self.data_loaded:
            self.load_data()

        v_measures = []
        for cluster_set in tqdm.tqdm(self.dataset[split], desc="Clustering"):
            evaluator = FlexibleClusteringEvaluator(
                cluster_set["sentences"],
                cluster_set["labels"],
                clustering_alg=self.clustering_alg,
                clustering_params=self.clustering_params.copy(),
                dim_red=self.dim_red,
                dim_red_params=self.dim_red_params.copy(),
            )

            metrics = evaluator(model)
            v_measures.append(metrics["v_measure"])

        v_mean = np.mean(v_measures)
        v_std = np.std(v_measures)
        return {"v_measure": v_mean, "v_measure_std": v_std}

    @dynamic_description
    def description(self):
        raise NotImplementedError


class BlurbsClusteringS2S(AbsTaskFlexibleClustering):
    @dynamic_description
    def description(self) -> dict:
        return {
            "name": "BlurbsClusteringS2S",
            "hf_hub_name": "slvnwhrl/blurbs-clustering-s2s",
            "description": "Clustering of book blurbs (titles only).",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": ["v_measure"],
        }


class BlurbsClusteringP2P(AbsTaskFlexibleClustering):
    @dynamic_description
    def description(self) -> dict:
        return {
            "name": "BlurbsClusteringP2P",
            "hf_hub_name": "slvnwhrl/blurbs-clustering-p2p",
            "description": "Clustering of book blurbs (titles + blurbs).",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": ["v_measure"],
        }


class TenKGnadClusteringS2S(AbsTaskFlexibleClustering):
    @dynamic_description
    def description(self) -> dict:
        return {
            "name": "TenKGnadClusteringS2S",
            "hf_hub_name": "slvnwhrl/tenkgnad-clustering-s2s",
            "description": "Clustering of German news articles titles.",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": ["v_measure"],
        }


class TenKGnadClusteringP2P(AbsTaskFlexibleClustering):
    @dynamic_description
    def description(self) -> dict:
        return {
            "name": "TenKGnadClusteringP2P",
            "hf_hub_name": "slvnwhrl/tenkgnad-clustering-p2p",
            "description": "Clustering of German news articles (titles + body).",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": ["v_measure"],
        }
