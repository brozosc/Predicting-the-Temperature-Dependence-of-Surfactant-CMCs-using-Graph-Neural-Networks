# -*- coding: utf-8 -*-
"""
@author: BrozosCh
"""
from smiles_to_molecules import MyOwnDataset
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch.nn import Sequential, Linear
from torch_scatter import scatter_add
import numpy as np
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd 
import numpy as np
from utils import EarlyStopping, error_metrics


# hyperparameters of the model
parser = argparse.ArgumentParser()
parser.add_argument('--plot', default = True) # Plot of train, validation and test error plots
parser.add_argument('--epochs', default=300)   # number of epochs
parser.add_argument('--dim', default=128)   # size of hidden node states
parser.add_argument('--lrate', default=0.005)   #  learning rate
parser.add_argument('--batch', default = 32)  # batch size
parser.add_argument('--split_type', default = 2) # Type of split described in the manuscript
parser.add_argument('--early_stopping_patience', default=60)   # number of epochs until early stopping
parser.add_argument('--lrfactor', default=0.8)   # decreasing factor for learning rate
parser.add_argument('--lrpatience', default=3)   # number of consecutive epochs without model improvement after which learning rate is decreased


args, unknow = parser.parse_known_args()
plot = args.plot
epochs = int(args.epochs)
dim = int(args.dim)
lrate = float(args.lrate)
batch = int(args.batch)
lrfactor = float(args.lrfactor)
lrpatience = int(args.lrpatience)
early_stopping_patience = int(args.early_stopping_patience)
split_type = int(args.split_type)

# The GNN model architecture.

class GNNReg(torch.nn.Module):
    def __init__(self):
        super(GNNReg, self).__init__() 
        self.lin0 = Linear(dataset.num_features, dim)         # Initial linear transformation layer for node features
            
        self.transformation_layer = torch.nn.Linear(dataset.num_edge_features, dim)  # Initial linear transformation layer for edge features
        gine_nn = Sequential(Linear(dim, int(dim*2)), nn.ReLU(), Linear(int(dim*2), dim))   
            
        self.conv1 = GINEConv(gine_nn, train_eps = False)  # The graph convolutinal layer   

        self.fc1 = torch.nn.Linear(dim+1, dim)     # Initial layer of the MLP. The input dimension is increased by 1 neuron, to incorporate the temperature information.  
        self.fc2 = torch.nn.Linear(dim, dim)
        self.fc3 = torch.nn.Linear(dim , 1)
        
    
    def forward(self, data):
        x, edge_index, edge_attr, temp  = data.x , data.edge_index, data.edge_attr, data.T
        x = F.relu(self.lin0(data.x))
        
        x = F.relu(self.conv1(x, edge_index, edge_attr = self.transformation_layer(data.edge_attr)))
        x_forward = x

        x = scatter_add(x_forward, data.batch, dim=0)
        temp = 10*temp  # The temperature was originally normalized between {0,1}. As described in our work, we found the normalization between {0,10} to perform better. Therefore, this extra step was added.

        x = torch.cat([x, temp.reshape(x.shape[0],1)], dim = 1) 
 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The minimum and maximum temperatures are neccessary values for temperature unscaling. The oroginal value is calculated as:  temp = scaled_value * (max_temp - min_temp) + min_temp`

min_temp = torch.tensor(273.15, dtype = torch.float32)
max_temp = torch.tensor(363.15, dtype = torch.float32) 


# Load the task's corresponding dataset

if split_type == 1:
    dataset = MyOwnDataset(root = r'split_type_1\Train')
    ext_test_dataset = MyOwnDataset(root = r"split_type_1\Test")
elif split_type == 2:
    dataset = MyOwnDataset(root = r'split_type_2\Train')
    ext_test_dataset = MyOwnDataset(root = r"split_type_2\Test")
else:
    print('Error in split type')

dataset.data.y = dataset.data.y[:,0]
ext_test_dataset.data.y = ext_test_dataset.data.y[:,0]

# Normalization of the target property

mean = torch.as_tensor(dataset.data.y, dtype=torch.float).mean()
std = torch.as_tensor(dataset.data.y, dtype=torch.float).std()
dataset.data.y = (dataset.data.y - mean) / std
ext_test_dataset.data.y = (ext_test_dataset.data.y - mean) / std



def data_preparation(seed):
    torch.manual_seed(seed)
    dataset.shuffle()
    val_dataset = dataset[:200]
    train_dataset = dataset[200:]
    train_loader = DataLoader(train_dataset, batch_size = batch, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch)
    test_loader = DataLoader(ext_test_dataset[:], batch_size = len(ext_test_dataset))
    return train_loader, val_loader, test_loader


def train(loader, model, optimizer):
    model.train()
    loss_all = abs_loss_all = total_examples = 0
    norm_train_mae, train_mae = 0, 0
    predicted_train, real_train = [],[]
    temp_train_predictions,temp_train_real = [], []
    
    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data)
        target = data.y.view(-1,1)

        loss = F.mse_loss(out, target)
        loss.backward()
        loss_all += loss * data.num_graphs
        total_examples += data.num_graphs
        optimizer.step()
        
        
        #Standardized errors calculation. They are calculated but not returned.
        norm_train_rmse = torch.sqrt(loss_all/total_examples)
        norm_train_mae += (out - target).abs().sum(0).item()  
        
        #calculating the unstandardized errors
        out_standardized = out*std + mean
        target_standardized = target*std + mean

        abs_loss = F.mse_loss(out_standardized, target_standardized)  # The MSE loss on the unstandardized data is calculated
        abs_loss_all += abs_loss*data.num_graphs
        
        temp_train_predictions.append(out_standardized)
        temp_train_real.append(target_standardized)
        
        train_rmse = torch.sqrt(abs_loss_all/total_examples)     
        train_mae += (out_standardized - target_standardized).abs().sum(0).item()

    
    predicted_train = torch.cat(temp_train_predictions, dim = 0)
    real_train = torch.cat(temp_train_real, dim = 0)
    
    _, _, train_mape = error_metrics(real_train.detach().numpy(), predicted_train.detach().numpy())


     #We report only the undstandardized errors   
    return abs_loss_all / len(loader.dataset),  train_rmse.item(), train_mae / len(loader.dataset)

def test(loader, model, optimizer):
    model.eval()
    val_mae, val_rmse = 0, 0
    loss_all_norm = norm_val_mae = abs_loss_all = 0 
    loss_all = total_examples = 0
    temp_test_predictions, temp_test_real = [],[]
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)

            target = data.y.view(-1,1)
        
            loss = F.mse_loss(out, target)
            loss_all += loss * data.num_graphs
            total_examples += data.num_graphs
       
            #calculating standardized errors

            norm_val_rmse = torch.sqrt(loss_all/total_examples)
            norm_val_mae += (out - target).abs().sum(0).item()
            
            #calculating the unstandardized errors
            out_standardized = out*std + mean
            target_data_standardized = target*std + mean

            abs_loss = F.mse_loss(out_standardized, target_data_standardized)
            abs_loss_all += abs_loss*data.num_graphs

            val_rmse = torch.sqrt(abs_loss_all/total_examples) 
            val_mae += (out_standardized - target_data_standardized).abs().sum(0).item()
            
            temp_test_predictions.append(out_standardized)
            temp_test_real.append(target_data_standardized)
            
            
    predicted_test = torch.cat(temp_test_predictions, dim = 0)
    real_test = torch.cat(temp_test_real, dim = 0)
    
    _, _, test_mape = error_metrics(real_test.detach().numpy(), predicted_test.detach().numpy())
            
    #We report only the unstandardized errors
    return abs_loss_all / total_examples , val_rmse.item(), val_mae / len(loader.dataset), test_mape*100

# Write predictions on the given dataset in an Excel file, together with the corresponding Smiles string. The results are been printed in an Excel File.

def write_predictions(loader, model, save_path, dataset_type,counter):
    model = model
    model.load_state_dict(torch.load(save_path+'base_model_{}.pt'.format(counter)))
    model.eval()
    smiles, predicted, measured, temperature = [],[],[], []
    df_exp = pd.DataFrame()
    pred, real_value, pred_list, rel_error = None, None,  [], []
    for data in loader:     
        for mol in data.smiles_id:
            smiles.append(mol)
        data = data.to(device)
        real_value = data.y*std + mean
        real_value = real_value.view(-1,1)

        pred = model(data)
        pred_norm = pred*std + mean

        real_diff = abs( (pred_norm - real_value) / real_value)
        normalized_temp = data.T *(max_temp - min_temp) + min_temp
        
        for k in pred_list:
            smiles.append(k[0])
            predicted.append(k[1])
            measured.append(k[2])
        for c in real_value:
            measured.append(c.item())
        for k in pred_norm:
            predicted.append(k.item())
        
        for d in real_diff:
            rel_error.append(d.item())
            
        for t in normalized_temp:
            temperature.append(round(t.item(),2))
            
    df_exp['SMILES'] = smiles
    df_exp['Predicted'] = predicted
    df_exp['Measured'] = measured
    df_exp['Relative_difference'] = rel_error
    df_exp['Temperature'] = temperature
    df_exp.to_excel(str(save_path)+str(dataset_type)+'_base_model_{}.xlsx'.format(counter), index = False)

#User defined path.
     
save_path = str('results\\')

    
# Function for saving the best model's parameters.

def save_checkpoint(model,filename):
    print('Saving checkpoint') 
    torch.save(model.state_dict(), save_path+filename)    
    time.sleep(1.5)

# This function returns the unstandardized predictions. Based on the seed, we can initiate it with different models.

def ensemble(loader, model, save_path, seed):
    model = model
    model.load_state_dict(torch.load(save_path+'base_model_{}.pt'.format(seed), map_location= torch.device('cpu')))
    model.eval()
    for data in loader:
        data = data.to(device)
        out = model(data)
        out_unstandardized = out*std + mean
    return out_unstandardized


# This function create different training-validation splits, initiates the model training, saves the model in the best epoch and can also save the predictions on the test set for ensemble predictions.
def training(counter):
    best_epoch, best_val_rmse , best_epoch_test_rmse = None, None, None
    best_val_mae, best_epoch_test_mae = None, None
    best_val_mape, best_epoch_test_mape = None, None
    
    train_losses, val_errors, test_errors = [], [],[]
    train_errors, val_losses, test_losses = [],[], []
    
    train_loader, validation_loader, test_loader = data_preparation(seed=counter)
    model = GNNReg().to(device)
    print(model)
    
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr = lrate)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=lrfactor, patience=lrpatience, min_lr=0.0000001)
     

    for epoch in range(1,epochs + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train_loss, train_rmse, train_mae = train(train_loader, model, optimizer)

        val_loss, val_rmse, val_mae, val_mape = test(validation_loader, model, optimizer)
        
        test_loss, test_rmse, test_mae, test_mape = test(test_loader, model, optimizer)

        print('Epoch: {} , Learning Rate: {}, Train error: {:.4f}, Val. error: {:.4f}, Test error: {:.4f}'.format(epoch, lr, train_loss, val_loss, test_loss))
        print('Val: val_rmse {:.4f}, val_mae {:.4f}, Test: test_rmse {:.4f}, test_mae {:.4f}'.format( val_rmse, val_mae, test_rmse, test_mae))
        
        scheduler.step(val_loss)
        train_losses.append(train_loss.detach().numpy())
        val_losses.append(val_loss.detach().numpy())
        test_losses.append(test_loss.detach().numpy())
        
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)
        test_errors.append(test_rmse)        
        
        if epoch > 0:
            if  best_val_rmse is None:
                best_epoch = epoch
                best_val_rmse, best_epoch_test_rmse = val_rmse, test_rmse
                best_val_mae, best_epoch_test_mae = val_mae, test_mae
                best_val_mape, best_epoch_test_mape =  val_mape, test_mape
            elif val_rmse < best_val_rmse:
                best_epoch = epoch
                best_val_rmse, best_epoch_test_rmse = val_rmse, test_rmse
                best_val_mae, best_epoch_test_mae = val_mae, test_mae
                best_val_mape, best_epoch_test_mape =  val_mape, test_mape
                save_checkpoint(model,'base_model_{}.pt'.format(counter))
        
            early_stopping(val_rmse)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    
    print('Best model with respect to validation error in epoch {:03d} with \nVal RMSE {:.5f}\nTest RMSE {:.5f}\n'.format(best_epoch, best_val_rmse, best_epoch_test_rmse))  
    
   # write_predictions(validation_loader, model, save_path, 'val', counter = counter)
   # write_predictions(test_loader,model, save_path, 'test', counter = counter)
 
   
    if plot is True:
        plt.title('Total loss')
        plt.plot(range(1,len(train_losses)+1), train_losses, label = 'Train')
        plt.plot(range(1,len(val_losses)+1), val_losses, label = 'Validation')
        plt.plot(range(1,len(test_losses)+1), test_losses, label='Test')
        plt.ylim(0,1.5)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()
               
        plt.title('RMSE')
        plt.plot(range(1,len(train_errors)+1), train_errors, label = 'Train')
        plt.plot(range(1,len(val_errors)+1), val_errors, label = 'Validation')
        plt.plot(range(1,len(test_errors)+1), test_errors, label='Test')
        plt.ylim(0,1.5)
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()
        
    else:
        pass
    return train_loss, val_rmse, val_mae, test_rmse, test_mae, best_val_rmse, best_epoch_test_rmse, best_val_mae, best_epoch_test_mae, val_mape, test_mape, best_val_mape, best_epoch_test_mape, val_loss, test_loss

# In this section, we define the number of runs we wish (40 during our work) and append the parameters to corresponding lists. Each run has a different training validation split.
val_rmse_40, test_rmse_40  = [], []
val_mae_40, test_mae_40 = [], []
val_mape_40, test_mape_40 = [], []
best_val_rmse_40, best_epoch_test_rmse_40 = [], []
best_val_mae_40, best_epoch_test_mae_40 = [], []
best_val_mape_40, best_epoch_test_mape_40 = [], []


def control_fun():
    for i in range(1, 41):
        out = training(i)
        test_rmse_40.append(out[3]) 
        test_mae_40.append(out[4])
        val_rmse_40.append(out[1])
        val_mae_40.append(out[2])
        best_val_rmse_40.append(out[5])
        best_epoch_test_rmse_40.append(out[6])
        best_val_mae_40.append(out[7])
        best_epoch_test_mae_40.append(out[8])
        val_mape_40.append(out[9])
        test_mape_40.append(out[10])
        best_val_mape_40.append(out[11])
        best_epoch_test_mape_40.append(out[12])



control_fun()

#Optionally the results are saved in a dataframe. 

df = pd.DataFrame()
df['val_rmse_40'] = val_rmse_40
df['val_mae_40'] =  val_mae_40
df['val_mape_40'] = val_mape_40
df['test_rmse_40'] = test_rmse_40
df['test_mae_40'] = test_mae_40
df['test_mape_40'] = test_mape_40
df['best_val_rmse_40'] = best_val_rmse_40
df['best_epoch_test_rmse_40'] = best_epoch_test_rmse_40
df['best_val_mae_40'] = best_val_mae_40
df['best_epoch_test_mae_40'] = best_epoch_test_mae_40
df['best_val_mape_40'] = best_val_mape_40
df['best_epoch_test_mape_40'] = best_epoch_test_mape_40
df['Learning_rate'] = lrate
df['Batch_size'] = batch
df.loc['mean'] = df.mean()