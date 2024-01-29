
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

    def normalization(self, data,min_value=0, max_value=1):
        # Normalize the input data
        min_val = np.min(data)
        max_val = np.max(data)

        scaled_data = min_value + (max_value - min_value) * (data - min_val) / (max_val - min_val)
        return scaled_data

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
        for file in file_list:
            input_path = data_path + '/' + file
            data = pd.read_csv(input_path)
            y = data.pop(str(data.columns[-1])).values

            data = torch.tensor(data.values, dtype=torch.float32)
            # data = self.normalization(data)
            
            y = torch.tensor(y, dtype=torch.float32)
            # y = self.normalization(y)

            for i in range(0, len(data) - seq_len):
                if i + seq_len > len(data):
                    break
                else:
                    _x = data[i:i + seq_len, :]
                    _y = y[i + seq_len-1]

                data_X.append(_x)
                data_Y.append(_y)

            return data_X, data_Y

        

    def data_loader(self, data_path, batch_size=32, seq_len=50, split=0.2):
        # Load and preprocess data, create data loaders for training and validation
        self.seq_len = seq_len
        data_X, data_Y = self.import_data(seq_len=self.seq_len, data_path=data_path)

        data_X = torch.FloatTensor(self.normalization(np.array(data_X)))
        data_Y = torch.FloatTensor(self.normalization(np.array(data_Y)))



        data_X = data_X.to(self.device)
        data_Y = data_Y.to(self.device)

        dataset = CustomDataset(data_X,data_Y)
        train_dataset, val_dataset = random_split(dataset, [int(len(data_X) * split), len(data_X) - int(len(data_X) * split)])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)

    def check_the_dataset(self):
        # Check the structure of the data loader
        print(self.train_loader)
        print(next(iter(self.train_loader)))

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def save_checkpoint(self, filename):
        # Save the model checkpoint
        torch.save(self.model, filename)

    def train(self, epochs):
        # Train the model for a specified number of epochs
        self.loss_log = []
        for epoch in tqdm(range(epochs)):
            total_train_loss = 0
            self.model.train()
            for inputs, labels in self.train_loader:
                self.model.reset_hidden_state()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                labels = torch.unsqueeze(labels,dim=1)
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
                    labels = torch.unsqueeze(labels,dim=1)
                    val_loss += self.criterion(outputs, labels).item()
                avg_val_loss = val_loss / len(self.val_loader)

            self.loss_log.append([avg_train_loss, avg_val_loss])

            print(f'Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
            self.save_checkpoint(os.path.join(self.pt_dir, f'model_epoch_{epoch}.pt'))



if __name__ == "__main__":
    try:
        # Hyperparameters
        batch_size = 4
        lr = 1e-4
        hidden_size = 20

        epochs = 500

        # File paths
        today = datetime.today()
        data_path = os.path.join('Data','2024_01_27', 'random_torque_vectoring')
        log_dir = os.path.join('Data/ML', str(today.date()), 'log')
        pt_dir = os.path.join('Data/ML', str(today.date()), "pt")

        # Create an instance of EstimationMy
        model = EstimationMy()

        # Load and preprocess data, create model, set training parameters, create directories, and train the model
        model.data_loader(data_path, batch_size=batch_size)
        model.model_setting(hidden_size=hidden_size)
        model.train_setting()
        model.make_dir(pt_dir=pt_dir,log_dir=log_dir)
        # model.check_the_dataset()
        model.train(epochs=epochs)

    except KeyboardInterrupt:
        print("Canceld by user...")