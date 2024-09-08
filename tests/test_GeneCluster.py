# Load the model to extract features to calculate correlations between genes
import torch
import torch as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append('../GeneCluster')
from utils import load_model

class ExpressionDataset(Dataset):
    '''
    custom Dataset class for classifier training.
    __len__: returns the number of samples in our dataset.
    __getitem__:  loads and returns a sample from the dataset at the given index idx.
    '''
    def __init__(self, data_path):
        '''
        label_file: label
        '''
        self.datas = pd.read_csv(data_path, sep='\t', index_col=0)    ##load datas
        self.datas = np.expand_dims(np.array(self.datas), axis=1)
        self.labels = np.zeros((len(self.datas)), dtype='float32')
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        label = self.labels[idx]
        data = torch.FloatTensor(data)
        label = torch.LongTensor(np.array(label))
        return data, label


def compute_features(dataloader, model, N):  
    model.eval()
    
    for batch, (input_tensor, _) in enumerate(dataloader): 
        with torch.no_grad():
            aux = model(input_tensor).data.cpu().numpy() 
        
        if batch == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')  

        aux = aux.astype('float32')  
        if batch < len(dataloader) - 1:  
            features[batch * 128: (batch + 1) * 128] = aux
        else:
            # special treatment for final batch 
            features[batch * 128:] = aux

    return features


if __name__ == "__main__":
    expr_data_path = 'normal_data.csv'
    expr_data = pd.read_csv(expr_data_path,sep = '\t',index_col=0)
    dataset = ExpressionDataset(expr_data_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=128,
                                            num_workers=8,
                                            pin_memory=True)
    
    model = load_model('../train_res/checkpoint.pth.tar')
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1]) 
    features = compute_features(dataloader, model, len(dataset))
    features = pd.DataFrame(features,index=expr_data.index)
    features.to_csv('./features.csv')
