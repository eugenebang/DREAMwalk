import argparse
import pickle
import numpy as np
from sklearn import metrics 

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from utils import set_seed, EarlyStopper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', type=str, required=True)
    parser.add_argument('--pair_file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', type=str, 
                default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--h_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_checkpoint', type=str, default='checkpoint.pt')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--validation_ratio', type=float, default=0.1)
    return parser.parse_args()


class mlp_model(nn.Module):
    def __init__(self,input_dim=128,h_dim=64):
        super(mlp_model,self).__init__()
        self.lin1=nn.Linear(input_dim,h_dim)
        self.lin2=nn.Linear(h_dim,1)
        
    def forward(self,x):
        x=F.relu(self.lin1(x))
        return torch.sigmoid(self.lin2(x))
    
def split_dataset(pairf, embeddingf, validr, testr, seed):
    with open(embeddingf,'rb') as fin:
        embedding_dict=pickle.load(fin)
    
    xs,ys=[],[]
    with open(pairf,'r') as fin:
        lines=fin.readlines()
        
    for line in lines[1:]:
        line=line.strip().split('\t')
        drug=line[0]
        dis=line[1]
        label=line[2]
        xs.append(embedding_dict[drug]-embedding_dict[dis])
        ys.append(int(label))
        
    # dataset split
    x,y = {},{}
    x['train'],x['test'],y['train'],y['test']=train_test_split(
        xs,ys,test_size=testr, random_state=seed, stratify=ys)
    if validr > 0:
        x['train'],x['valid'],y['train'],y['valid']=train_test_split(
            x['train'],y['train'],test_size=validr/(1-testr), random_state=seed, stratify=y['train'])
    else:
        x['valid'],y['valid']=[],[]
    return x, y

def predict_dda(embeddingf, pairf, seed, patience, device, 
               h_dim, lr, modelf, validr, testr):
    set_seed(seed)
    x,y=split_dataset(pairf, embeddingf,
                    validr, testr, seed)
    
    x_train = torch.tensor(np.array(x['train']), dtype=torch.float32)
    y_train = torch.tensor(y['train'], dtype=torch.float32)

    x_valid = torch.tensor(np.array(x['valid']), dtype=torch.float32)
    y_valid = torch.tensor(y['valid'], dtype=torch.float32)

    x_test = torch.tensor(np.array(x['test']), dtype=torch.float32)
    y_test = torch.tensor(y['test'], dtype=torch.float32)
    
    input_dim=x_test[0].shape[0]
    model=mlp_model(input_dim, h_dim)
    early_stopper = EarlyStopper(patience=patience,
            printfunc=print,verbose=False,path=modelf)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    
    # model training    
    model=model.to(device)
    x_train=x_train.to(device)
    y_train=y_train.to(device)
    x_valid=x_valid.to(device)
    y_valid=y_valid.to(device)
    x_test=x_test.to(device)
    y_test=y_test.to(device)

    epoch=1
    while True:
        model.train()
        batch_loss = 0.0
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_fn(y_pred.view_as(y_train), y_train)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
        losses.append(batch_loss)

        with torch.no_grad():
            model.eval()
            y_pred=model(x_valid)
            valid_loss= loss_fn(y_pred.view_as(y_valid), y_valid)
            y_pred=model(x_test)
            test_loss = loss_fn(y_pred.view_as(y_test), y_test)

        print(f'[Epoch {epoch:3}] Train loss:{loss.item():.3f}, Valid loss: {valid_loss.item():.3f}, Test loss: {test_loss.item():.3f}')

        early_stopper(valid_loss, model)
        if early_stopper.early_stop:
            print()
            break
        epoch+=1
    
    # model performance measurement
    model.load_state_dict(torch.load(modelf,map_location=device))
    model.eval()
    with torch.no_grad():
        y_pred=model(x_test)
        test_loss = loss_fn(y_pred.view_as(y_test), y_test)
    print(f'loaded best model "{modelf}", valid loss: {early_stopper.val_loss_min:.3f}, test loss: {test_loss.item():.3f}')

    final_test_out=y_pred.cpu().detach().numpy()
    fpr, tpr, _ = metrics.roc_curve(y['test'], final_test_out, pos_label=1)
    auroc = metrics.roc_auc_score(y['test'], final_test_out)
    precision, recall, _ = metrics.precision_recall_curve(y['test'], final_test_out)
    aupr = metrics.auc(recall, precision)
    output_prob=[1 if i>0.5 else 0 for i in final_test_out]
    acc=metrics.accuracy_score(y['test'], output_prob)
    f1 = metrics.f1_score(y['test'], output_prob)

    print(f'Best model performance: AUROC {auroc:.3f}, AUPR {aupr:.3f}, Acc {acc:.3f}, F1 {f1:.3f}')
    print('='*50)
          
if __name__ == '__main__':
    args=parse_args()
    args={'embeddingf':args.embedding_file,
     'pairf':args.pair_file,
     'seed':args.seed,
     'patience':args.patience,
     'device':args.device,
     'h_dim':args.h_dim,
     'lr':args.lr,
     'modelf':args.model_checkpoint,
     'testr':args.test_ratio,
     'validr':args.validation_ratio
     }
    predict_dda(**args)
