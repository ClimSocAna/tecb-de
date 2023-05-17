import argparse
import logging

from ClusteringTasks import (
    BlurbsClusteringP2P,
    BlurbsClusteringS2S,
    RedditClusteringP2P,
    RedditClusteringS2S,
    TenKGnadClusteringP2P,
    TenKGnadClusteringS2S,
)
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# logging.basicConfig(level=logging.INFO)

base_tasks = [
    BlurbsClusteringS2S,
    BlurbsClusteringP2P,
    TenKGnadClusteringS2S,
    TenKGnadClusteringP2P,
]

# change task config here
task_configs = [
    {"dim_red": None, "clustering_alg": "minibatch_kmeans"},
    {"dim_red": "pca", "clustering_alg": "minibatch_kmeans"},
    {"dim_red": "umap", "clustering_alg": "minibatch_kmeans"},
    {"dim_red": "pca+umap", "clustering_alg": "minibatch_kmeans"},
    {"dim_red": None, "clustering_alg": "agglomerative"},
    {"dim_red": "pca", "clustering_alg": "agglomerative"},
    {"dim_red": "umap", "clustering_alg": "agglomerative"},
    {"dim_red": "pca+umap", "clustering_alg": "agglomerative"},
    {"dim_red": "pca", "clustering_alg": "hdbscan"},
    {"dim_red": "umap", "clustering_alg": "hdbscan"},
    {"dim_red": "pca+umap", "clustering_alg": "hdbscan"},
    {"dim_red": "pca", "clustering_alg": "dbstream"},
    {"dim_red": "umap", "clustering_alg": "dbstream"},
    {"dim_red": "pca+umap", "clustering_alg": "dbstream"},
]

# change models here
model_names = [
    "deepset/gbert-base",
    "deepset/gbert-large",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/sentence-t5-xxl",
]


def main(args: argparse.ArugmentParser):
    if args.include_reddit:
        base_tasks.extend([RedditClusteringS2S, RedditClusteringP2P])
    for model_name in model_names:
        model = SentenceTransformer(model_name)
        evaluation = MTEB(
            tasks=[task(**config) for task in base_tasks for config in task_configs]
        )
        evaluation.run(model, output_folder=f"results/{model_name.split('/')[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-reddit", action="store_true")

    args = parser.parse_args()
    main(args)
