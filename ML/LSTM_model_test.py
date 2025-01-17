
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
import torch.onnx
from torch.profiler import profile, record_function, ProfilerActivity
from pthflops import count_ops
# Custom Dataset class for handling input data and labels
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# LSTM model class for estimationcond
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

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

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
        self.x_min_value =  np.array([-0.6, 0, 0, 0, 0, -30, -1, -10, 0])
        self.y_min_value = np.array([-3000,-3000,-3000,-3000])
        self.x_max_value =  np.array([0.6, 30, 30, 30, 30, 30, 1, 10, 35])
        self.y_max_value = np.array([3000,3000,3000,3000])

    def normalization(self,data,min_val,max_val):
        # Normalize the input data
        data_shape = data.shape
        data = np.reshape(data,(-1,data.shape[-1]))

        min_value = np.zeros(shape = np.size(min_val))
        max_value = np.ones(shape = np.size(max_val))
        scaled_data = min_value + (max_value - min_value) * (data - min_val) / (max_val - min_val)

        negative_indices = scaled_data < 0
        scaled_data[negative_indices] = 0

        over_indices = scaled_data > 1
        scaled_data[over_indices] = 1

        scaled_data = np.reshape(scaled_data,data_shape)
        return scaled_data
        
        
    def denormalization(self, scaled_data):

        min_val = self.y_min_value
        max_val = self.y_max_value

        denormalized_data = min_val + (max_val - min_val) * (scaled_data)

        return denormalized_data
    
    def check_output_with_denormalization(self, outputs, labels):
        # Convert the outputs and labels to numpy arrays
        outputs_np = outputs.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()

        # Denormalize the outputs and labels
        outputs_denorm = self.denormalization(outputs_np)
        labels_denorm = self.denormalization(labels_np)

        # Print the denormalized outputs and labels for comparison
        print("Denormalized Outputs:")
        print(outputs_denorm[0:10,:])
        print("Denormalized Labels:")
        print(labels_denorm[0:10,:])

        plt.plot(outputs_denorm,'r',labels_denorm,'b')
        # plt.show()
          

    def make_dir(self, log_dir, pt_dir, onnx_dir):
        # Create directories for logs and saved models
        self.pt_dir = os.path.join(pt_dir, str(self.hidden_size))
        self.log_dir = os.path.join(log_dir, str(self.hidden_size))
        self.onnx_dir = os.path.join(onnx_dir, str(self.hidden_size))
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.pt_dir, exist_ok=True)
        os.makedirs(self.onnx_dir, exist_ok=True)

    def import_data(self,seq_len,data_path):
        
        file_list = os.listdir(data_path)
        data_X = []
        data_Y = []
        print("importing Data Start")
        for idx, file in enumerate(file_list):
            input_path = data_path + '/' + file
            data = pd.read_csv(input_path)
            y = data.iloc[:, -4:].values
            data = data.iloc[:, :-4].values
            for i in range(0, len(data) - seq_len):
                _x = data[i:i + seq_len, :]
                _y = np.array([y[i + seq_len-1]])
                data_X.append(_x)
                data_Y.append(_y)
        return data_X, data_Y

        

    def data_loader(self, data_path, batch_size=32, seq_len=50, split=0.8):
        # Load and preprocess data, create data loaders for training and validation
        self.seq_len = seq_len
        self.batch_size = batch_size
        data_X, data_Y = self.import_data(seq_len=self.seq_len, data_path=data_path)
        self.np_x = np.array(data_X)
        self.np_y = np.array(data_Y)
        norm_x=  self.normalization(self.np_x,self.x_min_value,self.x_max_value)
        norm_y =  self.normalization(self.np_y,self.y_min_value,self.y_max_value)
        data_X = torch.FloatTensor(norm_x)
        data_Y = torch.FloatTensor(norm_y)


        data_X = data_X.to(self.device)
        data_Y = data_Y.to(self.device)

        dataset = CustomDataset(data_X,data_Y)
        train_dataset, val_dataset = random_split(dataset, [int(len(data_X) * split), len(data_X) - int(len(data_X) * split)])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)

        sample_batch = next(iter(self.train_loader))
        self.input_size = sample_batch[0].shape[2]

    def model_setting(self, hidden_size, output_size=1, num_layers=1):
        # Initialize the LSTM model
        self.hidden_size = hidden_size
        self.layer_size = num_layers
        self.output_size = output_size
        self.model = EstimationLSTM(self.input_size, self.hidden_size, self.output_size, self.seq_len, num_layers)
        self.check_model = EstimationLSTM(self.input_size, self.hidden_size, self.output_size, self.seq_len, num_layers)
    
        self.model.to(self.device)
        self.check_model.to('cpu')

    def train_setting(self, lr=1e-4, loss=nn.MSELoss()):
        # Set training parameters including learning rate and loss function
        self.criterion = loss
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def save_checkpoint(self, pt_name,onnx_name):
        # Save the model checkpoint
        torch.save(self.model, pt_name)
        dummy_input = torch.randn(self.batch_size,self.seq_len,self.input_size).to(device=self.device)
        torch.onnx.export(self.model,dummy_input,onnx_name,export_params=True,
                        opset_version= 10, 
                        input_names=['Modelinput'],
                        output_names=['Modeloutput'])
    def save_log(self,list):
        # List is the n*2 matrix list that contains logs
        logs =np.array(list)
        log_df = pd.DataFrame(logs,columns=["Test",'Validation'])
        log_df.to_csv(self.log_dir+'/log.csv')

    def import_pt_model(self,pt_path,hidden_size):
        self.hidden_size = hidden_size
        model_pt = torch.load(pt_path)
        check_pt = torch.load(pt_path)
        self.model = model_pt
        self.check_model = check_pt.to('cpu')

    def check_the_trained_model(self,pt_path):
        pt_list = os.listdir(os.path.join(pt_path,str(self.hidden_size)))
        pt_file = pt_list[-1]
        model_pt =torch.load(os.path.join(pt_path,str(self.hidden_size),pt_file),map_location=self.device)
        with torch.no_grad():
                    inputs, labels = next(iter(self.train_loader))
                    outputs = model_pt(inputs)
                    self.check_output_with_denormalization(outputs,labels)

    def computing_time_calculation(self,epoch):
        inputs = torch.randn(1,1,9)
        flops = count_ops(self.check_model,inputs)
        print(flops)
        with profile(activities=[ProfilerActivity.CPU],
                profile_memory=True, record_shapes=True) as prof:
            self.check_model(inputs)
        prof.export_chrome_trace(os.path.join(self.log_dir,'trace'+str(epoch)+'.json'))




    def train(self, epochs):
        # Train the model for a specified number of epochs
        self.loss_log = []
        for epoch in tqdm(range(epochs)):
            if epoch == 50:
                self.train_setting(lr=self.lr * 0.1)
            elif epoch == 200:
                self.train_setting(lr=self.lr * 0.1)
            elif epoch == 300:
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
                    val_loss += self.criterion(outputs, labels).item()
                avg_val_loss = val_loss / len(self.val_loader)

            self.loss_log.append([avg_train_loss, avg_val_loss])

            print(f'Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
            self.save_checkpoint(os.path.join(self.pt_dir, f'model_epoch_{epoch}.pt'),os.path.join(self.onnx_dir, f'model_epoch_{epoch}.onnx'))
            # self.computing_time_calculation(epoch)
        print("Train_end")

        self.save_log(self.loss_log)


if __name__ == "__main__":
    try:
        # Hyperparameters
        batch_size = 64
        lr = 1e-3
        seq_len = 200
        hidden = 80
        epochs = 800
        pretrained = False

        # File paths
        today = datetime.today()
        data_path = os.path.join('Data', 'Data_test')
        log_dir = os.path.join('Data/ML/Test', str(today.date()), 'log')
        pt_dir = os.path.join('Data/ML/Test', str(today.date()), "pt")
        onnx_dir = os.path.join('Data/ML/Test', str(today.date()), "onnx")


        pt_path = 'Data/ML/pretrained_model/H_40_FC_1.pt'
        
        if isinstance(hidden, list):
            for hidden_size in hidden:
                model = EstimationMy()
                # Load and preprocess data, create model, set training parameters, create directories, and train the model
                model.data_loader(data_path, seq_len=seq_len,batch_size=batch_size)
                if pretrained ==True:
                    model.import_pt_model(pt_path,hidden_size=hidden_size)
                else:
                    model.model_setting(hidden_size=hidden_size,output_size=4)
                model.train_setting()
                model.make_dir(pt_dir=pt_dir,log_dir=log_dir,onnx_dir = onnx_dir)
                model.train(epochs)            

        else:
            model = EstimationMy()
            # Load and preprocess data, create model, set training parameters, create directories, and train the model
            model.data_loader(data_path, seq_len=seq_len,batch_size=batch_size)
            if pretrained ==True:
                model.import_pt_model(pt_path,hidden_size=hidden)
            else:
                model.model_setting(hidden_size=hidden,output_size=4)
            model.train_setting()
            model.make_dir(pt_dir=pt_dir,log_dir=log_dir,onnx_dir = onnx_dir)
            model.train(epochs)

    except KeyboardInterrupt:
        print("Canceld by user...")