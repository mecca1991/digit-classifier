from torch import nn

class DigitClassifierModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.convnn = nn.Sequential(
            nn.Conv2d(1, 28, 3),
            nn.ReLU(),
            nn.Conv2d(28, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()
    
    def forward(self, input):
        output = self.convnn(input)
        flat = self.flatten(output)

        output = self.dropout(self.activation(self.fc1(flat)))
        output = self.dropout(self.activation(self.fc2(output)))
        output = self.fc3(output)
        return output