# GeneCluster

GeneCluster, a gene clustering method for Single Cell RNA-seq data.

## Requirements
python --- 3.8.10

scanpy --- 1.8.2

umap-learn --- 0.5.3

pytorch --- 1.8.1

torchvision --- 0.9.1

faiss-gpu --- 1.7.1

## Usage
The raw single-cell data in h5 format first goes through DataProcess.py to get the gene-cell in csv format, and the csv format is fed into the model for training.

'''
python main.py --nmb_cluster 20 --batch 128 --epochs 100 \
               --lr 0.01 --data_path normal_data.csv \
               --ckpt_path train_res &
'''

or

'''
sh train.sh
'''

Please note that the model outputs gene embeddings extracted by training (features.npy), which can be used to predict gene co-expression relationships and identify gene modules.

Here is an example on pbmc3k.h5ad scRNA-seq data.

## Contact
Yuting Bai (yutingya820@163.com)