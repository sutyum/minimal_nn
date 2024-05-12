#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Activation function (sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Neural network structure
typedef struct {
    int num_layers;
    int* layer_sizes;
    double*** weights;
    double** biases;
} NeuralNetwork;

// Create a new neural network
NeuralNetwork* create_network(int num_layers, int* layer_sizes) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->layer_sizes = (int*)malloc(num_layers * sizeof(int));
    nn->weights = (double***)malloc((num_layers - 1) * sizeof(double**));
    nn->biases = (double**)malloc((num_layers - 1) * sizeof(double*));

    for (int i = 0; i < num_layers; i++) {
        nn->layer_sizes[i] = layer_sizes[i];
    }

    for (int i = 0; i < num_layers - 1; i++) {
        nn->weights[i] = (double**)malloc(layer_sizes[i + 1] * sizeof(double*));
        nn->biases[i] = (double*)malloc(layer_sizes[i + 1] * sizeof(double));

        for (int j = 0; j < layer_sizes[i + 1]; j++) {
            nn->weights[i][j] = (double*)malloc(layer_sizes[i] * sizeof(double));
        }
    }

    // Initialize weights and biases randomly
    for (int i = 0; i < num_layers - 1; i++) {
        for (int j = 0; j < layer_sizes[i + 1]; j++) {
            for (int k = 0; k < layer_sizes[i]; k++) {
                nn->weights[i][j][k] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            }
            nn->biases[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }

    return nn;
}

// Feed forward through the neural network
void feed_forward(NeuralNetwork* nn, double* input, double* output) {
    double** activations = (double**)malloc(nn->num_layers * sizeof(double*));
    for (int i = 0; i < nn->num_layers; i++) {
        activations[i] = (double*)malloc(nn->layer_sizes[i] * sizeof(double));
    }

    // Copy input to the first layer activations
    for (int i = 0; i < nn->layer_sizes[0]; i++) {
        activations[0][i] = input[i];
    }

    // Feed forward through the layers
    for (int i = 1; i < nn->num_layers; i++) {
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            double sum = 0.0;
            for (int k = 0; k < nn->layer_sizes[i - 1]; k++) {
                sum += activations[i - 1][k] * nn->weights[i - 1][j][k];
            }
            sum += nn->biases[i - 1][j];
            activations[i][j] = sigmoid(sum);
        }
    }

    // Copy the last layer activations to the output
    for (int i = 0; i < nn->layer_sizes[nn->num_layers - 1]; i++) {
        output[i] = activations[nn->num_layers - 1][i];
    }

    // Free memory
    for (int i = 0; i < nn->num_layers; i++) {
        free(activations[i]);
    }
    free(activations);
}

// Train the neural network using backpropagation
void train(NeuralNetwork* nn, double* input, double* target, double learning_rate) {
    double** activations = (double**)malloc(nn->num_layers * sizeof(double*));
    double** errors = (double**)malloc(nn->num_layers * sizeof(double*));
    for (int i = 0; i < nn->num_layers; i++) {
        activations[i] = (double*)malloc(nn->layer_sizes[i] * sizeof(double));
        errors[i] = (double*)malloc(nn->layer_sizes[i] * sizeof(double));
    }

    // Feed forward
    feed_forward(nn, input, activations[nn->num_layers - 1]);

    // Calculate output layer errors
    for (int i = 0; i < nn->layer_sizes[nn->num_layers - 1]; i++) {
        errors[nn->num_layers - 1][i] = (target[i] - activations[nn->num_layers - 1][i]) * sigmoid_derivative(activations[nn->num_layers - 1][i]);
    }

    // Backpropagate errors
    for (int i = nn->num_layers - 2; i >= 0; i--) {
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            double error = 0.0;
            for (int k = 0; k < nn->layer_sizes[i + 1]; k++) {
                error += errors[i + 1][k] * nn->weights[i][k][j];
            }
            errors[i][j] = error * sigmoid_derivative(activations[i][j]);
        }
    }

    // Update weights and biases
    for (int i = 0; i < nn->num_layers - 1; i++) {
        for (int j = 0; j < nn->layer_sizes[i + 1]; j++) {
            for (int k = 0; k < nn->layer_sizes[i]; k++) {
                nn->weights[i][j][k] += learning_rate * errors[i + 1][j] * activations[i][k];
            }
            nn->biases[i][j] += learning_rate * errors[i + 1][j];
        }
    }

    // Free memory
    for (int i = 0; i < nn->num_layers; i++) {
        free(activations[i]);
        free(errors[i]);
    }
    free(activations);
    free(errors);
}

// Free memory allocated for the neural network
void free_network(NeuralNetwork* nn) {
    for (int i = 0; i < nn->num_layers - 1; i++) {
        for (int j = 0; j < nn->layer_sizes[i + 1]; j++) {
            free(nn->weights[i][j]);
        }
        free(nn->weights[i]);
        free(nn->biases[i]);
    }
    free(nn->weights);
    free(nn->biases);
    free(nn->layer_sizes);
    free(nn);
}

int main(int argc, char*argv[]) {
    if (argc != 2) {
        printf("Usage: %s <num_epochs>\n", argv[0]);
        return 1;
    }

    int num_epochs = atoi(argv[1]);

    // Define the layer sizes for a large neural network
    int num_layers = 3;
    int layer_sizes[] = {1000, 400, 200};

    // Create the neural network
    NeuralNetwork* nn = create_network(num_layers, layer_sizes);

    // Example training data
    double input[1000];
    double target[200];

    // Initialize input and target data (you would need to provide actual data)
    for (int i = 0; i < 1000; i++) {
        input[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    for (int i = 0; i < 200; i++) {
        target[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    // Train the network
    for (int i = 0; i < num_epochs; i++) {
        train(nn, input, target, 0.1);
    }

    // Test the trained network
    double output[200];
    feed_forward(nn, input, output);
    printf("Output:\n");
    for (int i = 0; i < 200; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    // Free the memory allocated for the network
    free_network(nn);

    return 0;
}
