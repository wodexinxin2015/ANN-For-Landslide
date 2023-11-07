# ----------------------------------------------------------------------------------------------------------------------
# Neural_Network_Class for the training of geotechnical data
# By Weijie Zhang, Hohai University
# ----------------------------------------------------------------------------------------------------------------------
import torch.nn as nn


# ----------------------------------------------------------------------------------------------------------------------
# Input --> Linear 1 --> Activation function 1 --> Linear 2 --> Activation function 2 --> Linear 3 --> Output
# ----------------------------------------------------------------------------------------------------------------------
class Neural_Network_Class(nn.Module):  # define the three-layer neural network
    # define the layer number of neural network
    def __init__(self, input_size, hidden_size, output_size, actifun_type1, actifun_type2):
        super(Neural_Network_Class, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        if actifun_type1 == 1:
            self.actfun1 = nn.Sigmoid()
        elif actifun_type1 == 2:
            self.actfun1 = nn.ReLU()
        elif actifun_type1 == 3:
            self.actfun1 = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        if actifun_type2 == 1:
            self.actfun2 = nn.Sigmoid()
        elif actifun_type2 == 2:
            self.actfun2 = nn.ReLU()
        elif actifun_type2 == 3:
            self.actfun2 = nn.Tanh()
        self.predict = nn.Linear(hidden_size, output_size)

    # define the forward propagation process
    def forward(self, input):
        out = self.linear1(input)
        out = self.actfun1(out)
        out = self.linear2(out)
        out = self.actfun2(out)
        out = self.predict(out)
        return out

# ----------------------------------------------------------------------------------------------------------------------
