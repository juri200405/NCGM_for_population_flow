import torch
import torch.nn as nn

class NCGM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hid_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 1)
        self.softmax = nn.Softmax(0)
    
    def forward(self, input):
        out = torch.tanh(self.fc1(input))
        out = self.fc2(out)
        out = self.softmax(out)
        return out

if __name__ == "__main__":
    model = NCGM(5, 8)
    input_tensor = torch.randn(3, 6, 8, 5)
    output_tensor = model(input_tensor)
    print(output_tensor)