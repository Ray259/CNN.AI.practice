from functions import *


class CNN:
    def __init__(self):
        self.conv1_filters = np.random.randn(3, 3)
        self.conv1_bias = np.random.randn(1)

        self.conv_activation = relu

        self.pool_size = (2, 2)

        self.fc_weights = np.random.randn(64, 10)
        self.fc_bias = np.random.randn(10)
        self.fc_activation = softmax

    def forward(self, input):
        # Conv 1
        conv1_output = self.conv_activation(
            convolution(input, self.conv1_filters, stride=1, padding=0) + self.conv1_bias)
        # Max pooling
        pooled_output = max_pooling(conv1_output, pool_size=self.pool_size)
        # Flatten
        flattened_output = pooled_output.flatten()
        # Fully connected
        fc_output = fully_connected(flattened_output, self.fc_weights, self.fc_bias)
        fc_output = self.fc_activation(fc_output)
        return fc_output
