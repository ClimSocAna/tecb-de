# German Text Embedding Clustering Benchmark

Shortcut: [Datasets](https://github.com/ClimSocAna/tecb-de#datasets) - [Results](https://github.com/ClimSocAna/tecb-de#results) - [Insallation](https://github.com/ClimSocAna/tecb-de#installation) - [Usage](https://github.com/ClimSocAna/tecb-de#usage)

## Remarks
This repository contains code to evaluate language models for clustering word embeddings as used in neural topic modelling (see for example [BERTopic](https://github.com/MaartenGr/BERTopic)) specifically for <b>German</b>. This work builds on [Massive Text Embedding Benchmark (MTEB)](https://github.com/embeddings-benchmark/mteb), which provides benchmark datasets and results for a wide range of tasks.

More specifically, this work contributes to mteb in the following ways:
- clustering datasets in German (MTEB only consider English datasets)
- the evaluation of more clustering algorithms


## Datasets

Currently, we provide 4 datasets. The datasets are built similarly to the English clustering datasets in MTEB. Unfortunately, there are fewer datasets available for German and, therefore, we were not able to build as many datasets (e.g. Arxiv only contains very few German papers). However, we plan to add more datasets in the future.

| **Name**              | **Hub URL**                      | **Description**                                              |
|-----------------------|----------------------------------|--------------------------------------------------------------|
| BlurbsClusteringS2S<br>([data ref.](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html)) | [slvnwhrl/blurbs-clustering-s2s](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-s2s/tree/main)   | Clustering of <b>book titles</b>: 17'726 unqiue samples, 28 splits with 177 to 16'425 samples and 4 to 93 unique classes. Splits are built similarly to MTEB's [ArxivClusteringS2S](https://huggingface.co/datasets/mteb/arxiv-clustering-s2s) ([Paper](https://arxiv.org/abs/2210.07316)). |
| BlurbsClusteringP2P<br>([data ref.](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html)) | [slvnwhrl/blurbs-clustering-p2p](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-p2p/tree/main)   | Clustering of <b>book blurbs</b> (title + blurb): Clustering of book titles: 18'084 unqiue samples, 28 splits with 177 to 16'425 samples and 4 to 93 unique classes. Splits are built similarly to MTEB's [ArxivClusteringP2P](https://huggingface.co/datasets/mteb/arxiv-clustering-sp2p) ([aper](https://arxiv.org/abs/2210.07316)). |
| TenKGNADClusteringS2S<br>([data ref.](https://ofai.github.io/million-post-corpus/)) | [slvnwhrl/tenkgnad-clustering-s2s](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-s2s) | Clustering of <b>news article titles</b>: 10'267 unique samples, 10 splits with 1'436 to 9'962 samples and 9 unique classes. Splits are built similarly to MTEB's [TwentyNewsgroupsClustering](https://huggingface.co/datasets/mteb/twentynewsgroups-clustering) ([Paper](https://arxiv.org/abs/2210.07316)).|
| TenKGNADClusteringP2P<br>([data ref.](https://ofai.github.io/million-post-corpus/)) | [slvnwhrl-tenkgnad-clustering-p2p](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-p2p) | Clustering of <b>news articles</b> (title + article body): 10'275 unique samples, 10 splits with 1'436 to 9'962 samples and 9 unique classes. Splits are built similarly to MTEB's [TwentyNewsgroupsClustering](https://huggingface.co/datasets/mteb/twentynewsgroups-clustering) ([Paper](https://arxiv.org/abs/2210.07316)). |

## Results
### k-means (same as MTEB)
| **Model** | **BlurbsClusteringS2S** | **BlurbsClusteringP2P** | **TenKGNADClusteringS2S** | **TenKGNADClusteringP2P** | **AVG**
|----|-------:|-------:|-------:|-------:|-------:|
| [deepset/gbert-base](https://huggingface.co/deepset/gbert-base) | 11.27 | 35.36 | 24.23 | 37.16 | 27.01 |
| [deepset/gbert-large](https://huggingface.co/deepset/gbert-large) | 13.34 | **39.30** | **34.97** | **41.69** | **32.33** |
| [T-Systems-onsite/cross-en-de-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer) | 12.69 | 30.81 | 10.94 | 23.5 | 19.49 |
| [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | **14.33** | 32.46 | 22.26 | 36.13 | 26.30 |

### additional clustering algorithms 
tbd
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
Simply run `python scripts/run_cteb_de.py`. This will produce an `results` folder. 


