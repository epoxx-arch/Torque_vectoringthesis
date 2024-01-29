import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from datetime import datetime
import torch.autograd.profiler as profiler
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

class DNN(nn.Module):
    def __init__(self,input_size,hidden_size,depth):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
    

    def forward(self,x):
        with profiler.record_function("DNN"):
            x = self.relu(self.fc1(x))
            for i in range(self.depth):
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
            x = self.relu(self.fc3(x))

        return x

class TV():
    def __init__(self):

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print('Cuda is available')
        else:
            print('there is no Cuda')
            self.device = torch.device('cpu')

        torch.cuda.set_device(self.device)

    def normalization(self, data,min_value=0, max_value=1):
        # Normalize the input data
        min_val = np.min(data)
        max_val = np.max(data)

        scaled_data = min_value + (max_value - min_value) * (data - min_val) / (max_val - min_val)
        return scaled_data

    def denormalization(self, scaled_data,original_data):

        min_val = np.min(original_data)
        max_val = np.max(original_data)

        denormalized_data = min_val + (max_val - min_val) * (scaled_data)

        return denormalized_data
    
    def check_output_with_denormalization(self, outputs, labels):
        # Convert the outputs and labels to numpy arrays
        outputs_np = outputs.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()

        # Denormalize the outputs and labels
        outputs_denorm = self.denormalization(outputs_np,self.y)
        labels_denorm = self.denormalization(labels_np,self.y)

        # Print the denormalized outputs and labels for comparison
        print("Denormalized Outputs:")
        print(outputs_denorm)
        print("Denormalized Labels:")
        print(labels_denorm)   


    def check_the_dataset(self):
        # Check the structure of the data loader
        print(next(iter(self.train_loader)))


    def data_loader(self,Data_path,batch_size = 32,split =0.2):
        Data = pd.read_csv(Data_path,header=0)
        
        y = Data.pop(str(Data.columns[-1])).values

        self.Data = np.array(Data.values, dtype=float)    
        Tensor_Data = torch.FloatTensor(self.normalization(self.Data))

        self.y = np.array(y, dtype=float)
        Tensor_y = torch.FloatTensor(self.normalization(self.y))

        Tensor_Data = Tensor_Data.to(self.device)
        Tensor_y = Tensor_y.to(self.device)


        X_train, X_val, y_train, y_val = train_test_split(Tensor_Data, Tensor_y, test_size=split, random_state=30)

        train_dataset = CustomDataset(X_train,y_train)
        val_dataset = CustomDataset(X_val,y_val)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    def make_dir(self,log_dir,pt_dir):
        self.pt_dir = os.path.join(pt_dir,str(self.hidden_size)+'_depth_'+str(self.depth))
        self.log_dir = log_dir
        if os.path.isdir(self.log_dir) == False:
            os.makedirs(self.log_dir)
        if os.path.isdir(self.pt_dir) == False:
            os.makedirs(self.pt_dir)
    
    def model_setting(self,hidden_size,depth=2):
        self.hidden_size = hidden_size
        self.depth = depth
        sample_batch = next(iter(self.train_loader))
        self.input_size = sample_batch[0].shape[1]
        print(self.input_size)
        self.model = DNN(self.input_size,self.hidden_size,self.depth)
        self.model.to(self.device)

    def train_setting(self,lr=1e-4, loss = nn.L1Loss()):
        self.criterion = loss
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr)

    def save_checkpoint(self,model, filename):
        torch.save(model, filename)

    def train(self,epoch):
        self.loss_log=[]
        
        for epoch in tqdm(range(epoch)):
            total_train_loss = 0
            self.model.train()
            if epoch == len(range(epoch)):
                self.lr = 0.1*self.lr
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                train_loss = self.criterion(outputs.squeeze(), labels)
                train_loss.backward()
                self.optimizer.step()
                total_train_loss += train_loss.item()
            avg_train_loss =total_train_loss/len(self.train_loader)

            # Validation loop
            self.model.eval()
            with torch.no_grad():
                val_loss =0
                for inputs, labels in self.val_loader:
                    outputs = self.model(inputs)
                    val_loss += self.criterion(outputs.squeeze(), labels).item()
                avg_val_loss = val_loss/ len(self.val_loader)
            
            self.loss_log.append([avg_train_loss,avg_val_loss])

            if epoch % 10 == 0 :
                with torch.no_grad():
                    inputs, labels = next(iter(self.train_loader))

                    with profiler.profile(with_stack=True, use_cuda= True,profile_memory= True) as prof:
                        outputs = self.model(inputs)
                    print(prof.key_averages().table(row_limit=1))
                    self.check_output_with_denormalization(outputs.squeeze(),labels)



            print(f'Epoch {epoch+1}, Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
            self.save_checkpoint(self.model,  os.path.join(self.pt_dir,f'model_epoch_{epoch}.pt'))
    
    def save_log(self):
        log = pd.DataFrame(self.loss_log, columns=['Train loss', 'Validation Loss'])
        with pd.ExcelWriter(os.path.join(self.log_dir,'log_hidden_'+str(self.hidden_size)+'_depth_'+str(self.depth)+'.xlsx'), engine="xlsxwriter") as writer:
             log.to_excel(writer, sheet_name="Loss")

    def del_model(self):
        del self.model
        torch.cuda.empty_cache()



if __name__ == "__main__":

    try:
        ## hyperparameters

        batch_size = 64
        lr = 1e-4
        Loss = nn.L1Loss()
        hidden_size = 10
        depth = 3
        epoch = 500
        #path
        today = datetime.today()
        Data_path = os.path.join('Data/ML','Custom_data','new_concat2.csv')
        log_dir = os.path.join('Data/ML',str(today.date()),'log') 
        pt_dir = os.path.join('Data/ML',str(today.date()),"pt")

        tv = TV()
        tv.data_loader(Data_path,batch_size=batch_size)
        tv.model_setting(hidden_size=hidden_size,depth=depth)
        tv.train_setting(lr=lr,loss=Loss)
        tv.make_dir(pt_dir=pt_dir,log_dir=log_dir)
        # tv.check_the_dataset()
        tv.train(epoch)
        tv.save_log()

    except KeyboardInterrupt:
        print("Canceld by user...")



                    


