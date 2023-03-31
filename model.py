import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, out_dim ):
        super(LSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(embedding_dim, out_dim)

        self.linear= nn.Linear(hidden_dim, out_dim)

    def forward(self, input):

        lstm_out, (ht, ct) = self.lstm(input)
        
        linear_out = self.linear(ht[-1])

        return linear_out


class Linear(nn.Module):
    #For binary classification
    def __init__(self,  hidden_dim, out_dim ):
        super(Linear, self).__init__()

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear= nn.Linear(hidden_dim, out_dim)
    def forward(self, input):
        n = nn.Sigmoid()
        linear_out = n(self.linear(input))
    
        return linear_out
   
    