clc; clear all;

%%
file_path = 'C:\Users\User\Desktop\Code\TV\Data\ML\ALL_data\TV_JM_141625.csv';
data_table = readtable(file_path);

data = table2array(data_table);

x = data(:,1:9);
y = data(:,10);

num_rows = height(x);
time_vector = (0:(num_rows-1)) * 0.1;
xt = timeseries(x,time_vector);
yt = timeseries(y,time_vector);
%%
A = sim("LSTM_import_smc.slx");
