# define a CNN Model class
import torch
from torch import nn

class CNN(torch.nn.Module):
    def __init__(self, conv_layers, activation_functions, maxpool_layers, fc_layers, loss_function, optimizer, use_dropout=False, lr=0.001):
        super(CNN, self).__init__()

        # Determine which activation function to use according to the input parameter activation_functions
        if activation_functions == "relu":
            self.activation = nn.ReLU()
        elif activation_functions == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_functions == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation function")

        # Determine which loss function to use according to the input parameter loss_function
        if loss_function == "mse":
            self.loss_function = nn.MSELoss()
        elif loss_function == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid loss function")

        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        for in_channels, out_channels, kernel_size in conv_layers:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))

        # Create maxpooling layers
        self.maxpool_layers = nn.ModuleList()
        for kernel_size in maxpool_layers:
            self.maxpool_layers.append(nn.MaxPool2d(kernel_size))

        # Create fully connected layers
        self.fc_layers = nn.ModuleList()
        for in_features, out_features in fc_layers:
            self.fc_layers.append(nn.Linear(in_features, out_features))


        # Determine which optimizer to use according to the input parameter optimizer
        self.lr = lr
        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Invalid optimizer")


        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):

        # Apply convolutional and maxpooling layers alternately
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.activation(x)
            x = self.maxpool_layers[i](x)

        # Flatten the output of the last maxpooling layer
        x = torch.flatten(x, 1)

        # Apply fully connected layers with activation except the last one
        for i in range(len(self.fc_layers) - 1):
            x = self.fc_layers[i](x)
            if self.use_dropout:
                x = self.dropout(x)
            x = self.activation(x)

        # Apply the last fully connected layer without activation
        x = self.fc_layers[-1](x)

        return x
