import torch.nn as nn

# https://github.com/udacity/deep-learning-v2-pytorch/blob/master/recurrent-neural-networks/time-series/Simple_GRU.ipynb
### Second, let's define the simple GRU model: many-to-many mapping/sequence-to-sequence
class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(GRU, self).__init__()

        self.model_name = 'gru'
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epoch = []   ## initial variable to record actual number during training
        self.train_err = []      ## initial variable to record actual training error
        self.validate_err = []

        # define an GRU with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)

        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)

        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        r_out, hidden = self.gru(x, hidden)

        # Need to generate final output of batch_size, seq_length, output_size
        output = self.fc(r_out)

        return output, hidden

    ## Convenient access method
    def getModelName(self):
        return (self.model_name)

    def getModelParams(self):
        return (self.hidden_dim, self.n_layers)

    def updateModel(self, epoch, train_err, validate_err):
        self.epoch.append(epoch)
        self.train_err.append(train_err)
        self.validate_err.append(validate_err)

    def getModelStats(self):
        return (self.epoch, self.train_err, self.validate_err)