# GeneCluster

GeneCluster, a deep learning based gene clustering method for single cell RNA-seq data.

## Requirements

python --- 3.8.10
numpy --- 1.20.3
numba --- 0.56.4
pandas --- 2.0.3
scanpy --- 1.9.3
h5py ---3.8.0
matplotlib ---3.7.3
sklearn --- 0.0
umap-learn --- 0.5.3
torch --- 1.8.1
torchvision --- 0.9.1
faiss-gpu --- 1.7.1

## Usage

The raw single-cell data in h5 format first goes through DataProcess.py to get the gene-cell in csv format, and the csv format is fed into the model for training.

## Example

```sh
# create env
conda create -n pytorch1.8
conda activate pytorch1.8
# After installing the required package, set parameters to run the model
cd GeneCluster/GeneCluster
python main.py --data_path path/to/dataset.csv --epochs 100 --nmb_cluster 20 --lr 0.01 --batch 128 --ckpt_path train_res   
```

or
```sh
sh train.sh
```


Please note that the model outputs gene embeddings extracted by training (features.npy), which can be used to predict gene co-expression relationships and identify gene modules.


## Contact

Yuting Bai (yutingya820@163.com)