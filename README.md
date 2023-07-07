# German Text Embedding Clustering Benchmark

Shortcut: [Datasets](https://github.com/ClimSocAna/tecb-de#datasets) - [Results](https://github.com/ClimSocAna/tecb-de#results) - [Insallation](https://github.com/ClimSocAna/tecb-de#installation) - [Usage](https://github.com/ClimSocAna/tecb-de#usage)

## Remarks
This repository contains code to evaluate language models for clustering word embeddings as used in neural topic modelling (see for example [BERTopic](https://github.com/MaartenGr/BERTopic)) specifically for <b>German</b>. This work builds on [Massive Text Embedding Benchmark (MTEB)](https://github.com/embeddings-benchmark/mteb), which provides benchmark datasets and results for a wide range of tasks.

More specifically, this work contributes to mteb in the following ways:
- clustering datasets in German (MTEB only consider English datasets)
- the evaluation of more clustering algorithms

:trophy: Note that you can contribute results to the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) as our datasets are officially part of MTEB (apart from the Reddit datasets, see below)! You can either use this library or MTEB directly to produce results. If you run into any problems, please raise an issue. :trophy:


## Datasets

Currently, we provide 4 datasets. The datasets are built similarly to the English clustering datasets in MTEB. Unfortunately, there are fewer datasets available for German and, therefore, we were not able to build as many datasets (e.g. Arxiv only contains very few German papers). However, we plan to add more datasets in the future.

| **Name**              | **Hub URL**                      | **Description**                                              |
|-----------------------|----------------------------------|--------------------------------------------------------------|
| BlurbsClusteringS2S<br>([data ref.](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html)) | [slvnwhrl/blurbs-clustering-s2s](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-s2s/tree/main)   | Clustering of <b>book titles</b>: 17'726 unqiue samples, 28 splits with 177 to 16'425 samples and 4 to 93 unique classes (as represented by genres, e.g. fantasy). On average, a sample is 23.17 chars long. Splits are built similarly to MTEB's [ArxivClusteringS2S](https://huggingface.co/datasets/mteb/arxiv-clustering-s2s) ([Paper](https://arxiv.org/abs/2210.07316)). |
| BlurbsClusteringP2P<br>([data ref.](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html)) | [slvnwhrl/blurbs-clustering-p2p](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-p2p/tree/main)   | Clustering of <b>book blurbs</b> (title + blurb): Clustering of book titles: 18'084 unqiue samples, 28 splits with 177 to 16'425 samples and 4 to 93 unique classes as represented by genres, e.g. fantasy. On average, a sample is 663.91 chars long. Splits are built similarly to MTEB's [ArxivClusteringP2P](https://huggingface.co/datasets/mteb/arxiv-clustering-sp2p) ([paper](https://arxiv.org/abs/2210.07316)). |
| TenKGNADClusteringS2S<br>([data ref.](https://ofai.github.io/million-post-corpus/)) | [slvnwhrl/tenkgnad-clustering-s2s](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-s2s) | Clustering of <b>news article titles</b>: 10'267 unique samples, 10 splits with 1'436 to 9'962 samples and 9 unique classes (as represented by, e.g. politics). On average, a sample is 50.97 chars long. Splits are built similarly to MTEB's [TwentyNewsgroupsClustering](https://huggingface.co/datasets/mteb/twentynewsgroups-clustering) ([paper](https://arxiv.org/abs/2210.07316)).|
| TenKGNADClusteringP2P<br>([data ref.](https://ofai.github.io/million-post-corpus/)) | [slvnwhrl-tenkgnad-clustering-p2p](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-p2p) | Clustering of <b>news articles</b> (title + article body): 10'275 unique samples, 10 splits with 1'436 to 9'962 samples and 9 unique classes (as represented by, e.g. politics). On average, a sample is 2648.46 chars long. Splits are built similarly to MTEB's [TwentyNewsgroupsClustering](https://huggingface.co/datasets/mteb/twentynewsgroups-clustering) ([paper](https://arxiv.org/abs/2210.07316)). |

### Reddit datasets

We also include two Reddit datasets in the benchmark (similar to MTEB's [RedditClustering](https://huggingface.co/datasets/mteb/reddit-clustering) and [RedditClusteringP2P](https://huggingface.co/datasets/mteb/reddit-clustering-p2p) datasets). However, we only provide ids, and if you want to use these datasets, you need to download the data yourself (see [Including the Reddit dataset](https://github.com/ClimSocAna/tecb-de#including-the-reddit-dataset) for instructions). The datasets contain "hot" and "top" submissions to 80 popular German subreddits and were extracted using [PRAW](https://praw.readthedocs.io/en/stable/).

| **Name**            | **Description** |
|---------------------|-----------------|
| RedditClusteringS2S | Clustering of reddit submission titles: 40'181 unique samples, 10 splits with 9'288 to 26'221 samples and 10 to 50 unique classes (as represented by subbredits, e.g. r/Finanzen). On average, a sample is 52.16 chars long. Splits are built similarly to MTEB's [RedditClustering](https://huggingface.co/datasets/mteb/reddit-clustering) ([paper](https://arxiv.org/abs/2210.07316)). |
| RedditClusteringP2P | Clustering of reddit submissions (title + body): 40'305 unique samples, 10 splits with 9'288 to 26'221 samples and 10 to 50 unique classes (as represented by subbredits, e.g. r/Finanzen). On average, a sample is 901.78 chars long. Splits are built similarly to MTEB's [RedditClusteringP2P](https://huggingface.co/datasets/mteb/reddit-clustering-p2p) ([paper](https://arxiv.org/abs/2210.07316)). |

***Important**: As of June 19, 2023 new [Data API Terms](https://www.redditinc.com/policies/data-api-terms) become effective for Reddit. Most likely, it will not be allowed anymore to use Reddit data for such purposes (see especially "2.4 User Content" in the terms). Make sure you understand these terms and use Reddit data accordingly.*

## Results
All results show the [V-measure](https://scikit-learn.org/stable/modules/clustering.html#homogeneity-completeness-and-v-measure) (multiplied by 100 and rounded to two decimal points).
### k-means (same as MTEB)
| **Model** | **BlurbsClusteringS2S** | **BlurbsClusteringP2P** | **TenKGNADClusteringS2S** | **TenKGNADClusteringP2P** | **RedditClusteringS2S** | **RedditClusteringP2P** | **AVG**|
|----|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| [deepset/gbert-base](https://huggingface.co/deepset/gbert-base) | 11.27 | 35.36 | 24.23 | 37.16 | 28.57 | 35.30 | 28.66 |
| [deepset/gbert-large](https://huggingface.co/deepset/gbert-large) | 13.34 | 39.30 | **34.97** | 41.69 | 34.35 | 44.61 | 34.71 |
| [deepset/gelectra-base](https://huggingface.co/deepset/gelectra-base) | 7.74 | 10.06 | 4.11 | 9.02 | 6.59 | 7.73 | 7.54 |
| [deepset/gelectra-large](https://huggingface.co/deepset/gelectra-large) |7.57 | 13.96 | 3.91 | 11.49 | 7.59 | 10.54 | 9.18 |
| [uklfr/gottbert-base](https://huggingface.co/uklfr/gottbert-base) |8.37 |  34.49 | 9.34 | 33.66 | 16.07 | 19.46 | 20.23 |
| [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | 14.33 | 32.46 | 22.26 | 36.13 |33.33 | 44.59 | 30.52 |
| [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) |15.81 | 34.38 | 22.00 | 35.96 | 36.39 | 48.43 | 32.16 |
| [T-Systems-onsite/cross-en-de-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer) | 12.69 | 30.81 | 10.94 | 23.50 | 27.98 | 33.01 | 23.16 |
| [sentence-transformers/use-cmlm-multilingual](https://huggingface.co/sentence-transformers/use-cmlm-multilingual) |15.24 | 29.63 | 25.64 | 37.10 | 33.62 | 49.70 | 31.82 |
| [sentence-transformers/sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base) |11.57 | 30.59 | 18.11 | **44.88** |  31.99 | 45.80|  30.49 |
| [sentence-transformers/sentence-t5-xxl](https://huggingface.co/sentence-transformers/sentence-t5-xxl) | **15.94** | **39.91** | 19.69 | 43.43 | **38.54** | **55.90** | **35.57** |
| [xlm-roberta-large](https://huggingface.co/xlm-roberta-large) |7.29 | 29.84 | 6.16 | 32.46 | 10.19 | 23.50 | 18.24 |

### additional clustering algorithms 
In addition to [k-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), we evaluate the following different clustering algorithms:
- [Agglomerative clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn-cluster-agglomerativeclustering) (distance-based, number of clusers is assumed)
- [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/) (density-based, number of clusters is assumed)
- [DBSTREAM](https://riverml.xyz/dev/api/cluster/DBSTREAM/#dbstream) (streaming algorithm, number of clusters is *not* assumed)

Inspired by [BERTopic](https://github.com/MaartenGr/BERTopic), we also evaluate [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn-decomposition-pca) and [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) as a "preprocessing" step.

*If you want add/evaluate more algorithms, please have a look at [FlexibleClusteringEvaluator.py](https://github.com/ClimSocAna/tecb-de/blob/main/scripts/FlexibleClusteringEvaluator.py) on how to achieve that.*


### UMAP + {k-means, HDBSCAN, DBSTREAM}
*For all results have a look at [results/tecb-de-full-results.csv](https://github.com/ClimSocAna/tecb-de/blob/3364f94faba7b235c7498a2bb724324064ac4537/results/tecb-de-full-results.csv).*
 
| **Model**  | **Algorithm**                    | **BlurbsClusteringS2S** | **BlurbsClusteringP2P** | **10KGNADClusteringS2S** | **10KGNADClusteringP2P** | **RedditClusteringS2S** | **RedditClusteringP2P** | **AVG** |
|------------|----------------------------------|------------------------:|------------------------:|-------------------------:|-------------------------:|--------:|--------:|--------:|
| [deepset/gbert-base](https://huggingface.co/deepset/gbert-base) | k-means<br> HDBSCAN<br> DBSTREAM | 12.81<br> 14.31<br> 12.70 | **38.81**<br> 22.83<br> 37.06 | 29.31<br> 05.44<br> **28.92**  | 43.61<br> 32.45 <br>  42.74 | 31.77<br> 17.21<br> 31.70 | 46.06<br> 31.99<br> 44.84 | 33.73<br> 20.71<br> **32.99** |
| [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | k-means<br> HDBSCAN<br> DBSTREAM | 13.80<br> **20.00**<br> 15.77 | 34.16<br> 30.67<br> 32.47 | 25.22<br> 24.90<br> 26.44 | **43.75**<br> 35.53<br> 41.31 | 32.64<br> 32.23<br> **33.14** | **47.46**<br> 39.70<br> 46.47 | 32.84<br> 30.51<br> 32.60 |


## Installation
If you want to run the code from this repository (for creating the Reddit dataset or model evaluation), clone this repository and move to the downloaded folder

```
git clone https://github.com/ClimSocAna/tecb-de.git 
```

and create a new environment with the necessary packages
```
python -m venv tecb-de

source tecb-de/bin/activate # Linux, MacOS
venv\Scripts\activate.bat # Windows

pip install -r requirements.txt
```

## Usage
### Running the evaluation
Simply run `python scripts/run_cteb_de.py`. This will produce an `results` folder. You can modify the script to run the evaluation for models and clustering algorithms (and configuration) of your choosing.

### Including the Reddit dataset
If you want to use the reddit dataset, you first have to download the data
```
# move to the reddit_data folder in tecb-de
# make sure you have PRAW and tdqm installed: pip install praw, pip install tqdm

# downloads the data and saves it to submissions.tsv
python download.py
```
Note that for this to work, you have to edit the `reddit_data/praw.ini` with your client data. You can find instructions [here](https://praw.readthedocs.io/en/stable/getting_started/authentication.html).

Then you can create the datasets
```
# creates the splits for both tasks (RedditClusteringS2S and ReddictClusteringP2P)
# and saves them in the reddit_data folder
python create_splits.py
```

Finally, you can can run the evaluation using the `--include-reddit` flag
```
# assuming your position is in the top-level folder
python scripts/run_cteb_de.py --reddit-flag
```

### Adaptive pre-training
If you want to experiment with adaptive pre-training, you can have a look at `scripts/run_fine_tuned_cteb_de.py`. Basically, it allows you to train models using whole word masking (WWM) and [TSDAE](https://arxiv.org/abs/2104.06979) and to evaluate on a clustering algorithm during training.

