import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        layers = []
        for i in range(len(layer_sizes) - 1):
            if i ==0:
                layers.append(nn.Linear(input_size, layer_sizes[0]))
            else:
                layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-2],layer_sizes[-1]))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def train_model(self, inputs, targets, epochs=1000, learning_rate=0.1,criterion = nn.MSELoss()):
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):

            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}: Loss {loss.item()}")

    def predict(self, test_data):
        with torch.no_grad():
            test_data = torch.from_numpy(test_data)
            predictions = self(test_data)
            return predictions.round().detach().numpy()

