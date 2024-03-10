// lstm.h
#ifndef LSTM_H
#define LSTM_H

// Assuming a simple LSTM model with fixed-size input, output, and hidden states for demonstration purposes.
// You will need to adjust these sizes based on your actual model.
#define INPUT_SIZE 9
#define HIDDEN_SIZE 30
#define OUTPUT_SIZE 1

// Hard-coded weights and biases for the LSTM.
// These arrays need to be filled with the actual values extracted from your PyTorch model.
// Example format (values should be replaced with your model's parameters):
static float W_ih[INPUT_SIZE * HIDDEN_SIZE] = {/* weights from input to hidden layer */};
static float W_hh[HIDDEN_SIZE * HIDDEN_SIZE] = {/* weights for hidden layer */};
static float b_ih[HIDDEN_SIZE] = {/* bias for input to hidden layer */};
static float b_hh[HIDDEN_SIZE] = {/* bias for hidden layer */};

#endif // LSTM_H