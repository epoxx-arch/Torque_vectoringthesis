clc; clear all
addpath("C:\Users\User\Desktop\Code\TV\Data\ML\Test\2024-04-19\onnx\80\")

Pytorch_net = importONNXNetwork("model_epoch_799.onnx","TargetNetwork","dlnetwork");
numFeatures = 9;
numHidden = 80;

Layers = Pytorch_net.Layers;

Temp_lstm = lstmLayer(numHidden,"CellState",zeros(numHidden,1),"HiddenState",zeros(numHidden,1));
Temp_lstm.InputWeights = Pytorch_net.Layers(6).InputWeights;
Temp_lstm.RecurrentWeights = Pytorch_net.Layers(6).RecurrentWeights;
Temp_lstm.Bias = Pytorch_net.Layers(6).Bias;
Temp_Fully1 = fullyConnectedLayer(numHidden,"Weights",Pytorch_net.Layers(10).Weights,"Bias",Pytorch_net.Layers(10).Bias);
Temp_Fully2 = fullyConnectedLayer(4,"Weights",Pytorch_net.Layers(11).Weights,"Bias",Pytorch_net.Layers(11).Bias);


Temp_layers = [
    sequenceInputLayer(numFeatures)
    Temp_lstm
    Temp_Fully1
    Temp_Fully2
    regressionLayer];

New_net = SeriesNetwork(Temp_layers);

