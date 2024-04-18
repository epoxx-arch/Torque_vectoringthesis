clc; clear all
addpath("C:\Users\User\Desktop\Code\TV\Data\ML\2024-04-16\onnx\80\")

Pytorch_net = importONNXNetwork("model_epoch_399.onnx","TargetNetwork","dlnetwork");
numFeatures = 6;
numHidden = 80;

Layers = Pytorch_net.Layers;

Temp_lstm = lstmLayer(numHidden,"CellState",zeros(numHidden,1),"HiddenState",zeros(numHidden,1));
Temp_lstm.InputWeights = Pytorch_net.Layers(6).InputWeights;
Temp_lstm.RecurrentWeights = Pytorch_net.Layers(6).RecurrentWeights;
Temp_lstm.Bias = Pytorch_net.Layers(6).Bias;
Temp_Fully = fullyConnectedLayer(1,"Weights",Pytorch_net.Layers(10).Weights,"Bias",Pytorch_net.Layers(10).Bias);


Temp_layers = [
    sequenceInputLayer(numFeatures)
    Temp_lstm
    Temp_Fully
    regressionLayer];

New_net = SeriesNetwork(Temp_layers);

