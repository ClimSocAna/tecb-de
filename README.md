# German Text Embedding Clustering Benchmark (tecb-de)

This repository contains code to evaluate language models for clustering word embeddings as used in neural topic modelling (see for example [BERTopic](https://github.com/MaartenGr/BERTopic)) specifically for <b>German</b>. This work builds on [Massive Text Embedding Benchmark (MTEB)](https://github.com/embeddings-benchmark/mteb), which provides benchmark datasets and results for a wide range of tasks.

More specifically, this work contributes to mteb in the following ways:
- clustering datasets in German (mteb only consider English datasets)
- the evaluation of more clustering algorithms


## Datasets

Currently, we provide 4 datasets:

* [BlurbsClusteringS2S](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-s2s), [BlurbsClusteringP2P](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-p2p/tree/main)
* [10kGNADClusteringS2S](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-s2s), [10kGNADClusteringP2P](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-p2p)


## Installation
If you want to run the code from this repository (for creating the Reddit dataset or model evaluation), clone this repository and move to the downloaded folder

```
git clone ... 
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


