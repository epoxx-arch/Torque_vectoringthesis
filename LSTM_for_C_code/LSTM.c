
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lstm.h"

// Define a structure for the LSTM state.
typedef struct {
    float h[HIDDEN_SIZE]; // Hidden state
    float c[HIDDEN_SIZE]; // Cell state
} LSTMState;

// Sigmoid activation function.
static inline float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

// Hyperbolic tangent activation function.
static inline float tanh_activation(float x) {
    return tanh(x);
}

// LSTM forward pass.
void lstm_forward(float input[INPUT_SIZE], LSTMState *state) {
    float i_gate[HIDDEN_SIZE], f_gate[HIDDEN_SIZE], o_gate[HIDDEN_SIZE], g_gate[HIDDEN_SIZE];
    float new_c[HIDDEN_SIZE], new_h[HIDDEN_SIZE];

    // Input, forget, output, and gate computations.
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        float input_sum = 0, hidden_sum = 0;
        for (int j = 0; j < INPUT_SIZE; ++j) {
            input_sum += W_ih[i * INPUT_SIZE + j] * input[j];
        }
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            hidden_sum += W_hh[i * HIDDEN_SIZE + j] * state->h[j];
        }

        float total_input = input_sum + hidden_sum + b_ih[i];
        float total_hidden = hidden_sum + b_hh[i];

        i_gate[i] = sigmoid(total_input);
        f_gate[i] = sigmoid(total_input + total_hidden); // Modify as per actual logic
        o_gate[i] = sigmoid(total_input);
        g_gate[i] = tanh_activation(total_input);
        
        // Cell state update.
        new_c[i] = f_gate[i] * state->c[i] + i_gate[i] * g_gate[i];
        
        // Hidden state update.
        new_h[i] = o_gate[i] * tanh_activation(new_c[i]);
    }

    // Update the LSTM state.
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        state->h[i] = new_h[i];
        state->c[i] = new_c[i];
    }
}

// Function to initialize the LSTM state.
void init_lstm_state(LSTMState *state) {
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        state->h[i] = 0.0f;
        state->c[i] = 0.0f;
    }
}