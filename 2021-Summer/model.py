import torch
import torch.nn as nn

class MLP_Classifier(nn.Module):
    def __init__(self, hidden_dim=256):
        super(MLP_Classifier, self).__init__()

        self.input_layer = nn.Linear(73, hidden_dim)
        self.hidden_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 50)

        self.activate = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.activate(self.input_layer(x))
        
        x = self.activate(self.hidden_layer1(x))
        x = self.dropout1(x)

        x = self.activate(self.hidden_layer2(x))
        x = self.dropout2(x)

        x = self.output_layer(x)

        return x

