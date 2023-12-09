import torch
import torch.nn as nn
from torch.autograd import Variable

class LedvoLSTM(nn.Module):
    def __init__(
        self, 
        input_size, 
        output_size, 
        hidden_size, 
        num_layers=1, 
        batch_size = 1024, 
        dropout=0.0
    ):
        super(LedvoLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
                
        
    def init_hidden(self, batch_size):
        # first is the hidden h
        # second is the cell c
        if torch.cuda.is_available():
            # print('use gpu')
            # print(self.device)
            return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()),
                    Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()))
        else:
            # print('use cpu')
            # print(self.device)
            return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)))


    def forward(self, x):
        # LSTM layers
        x, _ = self.lstm(x, self.hidden)

        # Fully connected layer
        x = self.fc(x[:, -1, :])  # Take the output from the last time step

        return x