
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt 

# Custom Dataset class for handling input data and labels
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# LSTM model class for estimation
class EstimationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len, num_layers):
        super(EstimationLSTM, self).__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        # LSTM layer followed by a fully connected layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through LSTM and fully connected layer
        with profiler.record_function("LSTM"):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1])
        return out

    def reset_hidden_state(self):
        # Reset hidden state of the LSTM
        self.hidden = (
            torch.zeros(self.num_layers, self.seq_len, self.hidden_size),
            torch.zeros(self.num_layers, self.seq_len, self.hidden_size)
        )

# Class for handling the entire training process
class EstimationMy:
    def __init__(self):
        # Check for CUDA availability and set the device accordingly
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print('CUDA is available')
        else:
            print('No CUDA, using CPU')
            self.device = torch.device('cpu')

        torch.cuda.set_device(self.device)

    def normalization(self,data,indices_tag):
        # Normalize the input data
        data_shape = data.shape
        data = np.reshape(data,(-1,data.shape[-1]))
        min_val = np.min(data,axis=0)
        max_val = np.max(data,axis=0)
        min_value = np.zeros(shape = np.size(min_val))
        max_value = np.ones(shape = np.size(max_val))
        indices = indices_tag + ' Min_Value :' + str(min_val) + 'Max_Value : ' + str(max_val) +'\n'
        scaled_data = min_value + (max_value - min_value) * (data - min_val) / (max_val - min_val)
        scaled_data = np.reshape(scaled_data,data_shape)
        return scaled_data, indices
        
        
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
        outputs_denorm = self.denormalization(outputs_np,self.np_y)
        labels_denorm = self.denormalization(labels_np,self.np_y)

        # Print the denormalized outputs and labels for comparison
        print("Denormalized Outputs:")
        print(outputs_denorm[0:10,:])
        print("Denormalized Labels:")
        print(labels_denorm[0:10,:])

        plt.plot(outputs_denorm,'r',labels_denorm,'b')
        # plt.show()
          

    def make_dir(self, log_dir, pt_dir):
        # Create directories for logs and saved models
        self.pt_dir = os.path.join(pt_dir, str(self.hidden_size))
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.pt_dir, exist_ok=True)

    def import_data(self,seq_len,data_path):
        
        file_list = os.listdir(data_path)
        data_X = []
        data_Y = []
        print("importing Data Start")
        for idx, file in enumerate(file_list):
            input_path = data_path + '/' + file
            data = pd.read_csv(input_path)
            y = data.pop(str(data.columns[-1])).values
            
            data = torch.tensor(data.values, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            
            for i in range(0, len(data) - seq_len):
                _x = data[i:i + seq_len, :]
                _y = np.array([y[i + seq_len-1]])
                data_X.append(_x)
                data_Y.append(_y)

        return data_X, data_Y

        

    def data_loader(self, data_path,indices_path, batch_size=32, seq_len=50, split=0.8):
        # Load and preprocess data, create data loaders for training and validation
        self.seq_len = seq_len
        data_X, data_Y = self.import_data(seq_len=self.seq_len, data_path=data_path)
        self.np_x = np.array(data_X)
        self.np_y = np.array(data_Y)
        norm_x, indices_x =  self.normalization(self.np_x,'x')
        norm_y, indices_y =  self.normalization(self.np_y,'y')
        data_X = torch.FloatTensor(norm_x)
        data_Y = torch.FloatTensor(norm_y)



        data_X = data_X.to(self.device)
        data_Y = data_Y.to(self.device)

        dataset = CustomDataset(data_X,data_Y)
        train_dataset, val_dataset = random_split(dataset, [int(len(data_X) * split), len(data_X) - int(len(data_X) * split)])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        os.makedirs(indices_path, exist_ok=True)
        with open(indices_path+'indices.csv','w') as file:
            file.write(indices_x+indices_y)


    def check_the_dataset(self):
        # Check the structure of the data loader
        print(len(self.train_loader))
        # print(next(iter(self.train_loader)))

    def model_setting(self, hidden_size, output_size=1, num_layers=1):
        # Initialize the LSTM model
        self.hidden_size = hidden_size
        sample_batch = next(iter(self.train_loader))
        self.input_size = sample_batch[0].shape[2]
        self.model = EstimationLSTM(self.input_size, self.hidden_size, output_size, self.seq_len, num_layers)
        self.model.to(self.device)

    def train_setting(self, lr=1e-4, loss=nn.MSELoss()):
        # Set training parameters including learning rate and loss function
        self.criterion = loss
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def save_checkpoint(self, filename):
        # Save the model checkpoint
        torch.save(self.model, filename)

    def check_the_trained_model(self,pt_path):
        pt_list = os.listdir(os.path.join(pt_path,str(self.hidden_size)))
        pt_file = pt_list[-1]
        model_pt =torch.load(os.path.join(pt_path,str(self.hidden_size),pt_file),map_location=self.device)
        with torch.no_grad():
                    inputs, labels = next(iter(self.train_loader))
                    outputs = model_pt(inputs)
                    self.check_output_with_denormalization(outputs,labels)
    def train(self, epochs):
        # Train the model for a specified number of epochs
        self.loss_log = []
        for epoch in tqdm(range(epochs)):
            if epoch % 100 == 0:
                self.train_setting(lr=self.lr * 0.1)
            elif epoch == 400:
                self.train_setting(lr=self.lr * 0.1)
            total_train_loss = 0
            self.model.train()
            for inputs, labels in self.train_loader:
                self.model.reset_hidden_state()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # labels = torch.unsqueeze(labels,dim=1)
                train_loss = self.criterion(outputs, labels)
                train_loss.backward()
                self.optimizer.step()
                total_train_loss += train_loss.item()

            avg_train_loss = total_train_loss / len(self.train_loader)

            # Validation loop
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for inputs, labels in self.val_loader:
                    outputs = self.model(inputs)
                    # labels = torch.unsqueeze(labels,dim=1)
                    val_loss += self.criterion(outputs, labels).item()
                avg_val_loss = val_loss / len(self.val_loader)
            # Check the test Result
            if epoch % 50 == 0 :
                with torch.no_grad():
                    inputs, labels = next(iter(self.train_loader))
                    outputs = self.model(inputs)
                    self.check_output_with_denormalization(outputs,labels)

            self.loss_log.append([avg_train_loss, avg_val_loss])

            print(f'Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
            self.save_checkpoint(os.path.join(self.pt_dir, f'model_epoch_{epoch}.pt'))



if __name__ == "__main__":
    try:
        # Hyperparameters
        batch_size = 512
        lr = 1e-3
        hidden_size = 10

        epochs = 1000

        # File paths
        today = datetime.today()
        data_path = os.path.join('Data','ML', 'ALL_data')
        log_dir = os.path.join('Data/ML', str(today.date()), 'log')
        pt_dir = os.path.join('Data/ML', str(today.date()), "pt")
        Normalization_dir = os.path.join('Data/ML', str(today.date()), "Normalization_value")
        
        # Create an instance of EstimationMy
        model = EstimationMy()

        # Load and preprocess data, create model, set training parameters, create directories, and train the model
        model.data_loader(data_path,Normalization_dir, batch_size=batch_size)
        # model.check_the_dataset()
        model.model_setting(hidden_size=hidden_size)
        model.train_setting()
        model.make_dir(pt_dir=pt_dir,log_dir=log_dir)
        
        
        # model.check_the_trained_model(pt_dir)
        
        model.train(epochs=epochs)

    except KeyboardInterrupt:
        print("Canceld by user...")