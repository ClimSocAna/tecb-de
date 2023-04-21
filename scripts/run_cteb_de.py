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

model_names = [
    "deepset/gbert-base",
    "deepset/gbert-large",
    "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

for model_name in model_names:
    model = SentenceTransformer(model_name)
    evaluation = MTEB(
        tasks=[task(**config) for task in base_tasks for config in task_configs]
    )
    evaluation.run(model, output_folder=f"results/{model_name.split('/')[-1]}")
