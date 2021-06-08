import torch
import torch.nn as nn
import torch.nn.functional as F

## It lays the neural network structure: how many layers and how many nodes within each layer 
# In the Class SimpleNet, it enables input_dim and hidden_dim to be input information
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
        
        # define all layers, here
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(0.3) #this is used to prevent overfitting, as nodes were dropped out randomly during the training
        self.sig = nn.Sigmoid()
        
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        # your code, here
        out = F.relu(self.fc1(x)) 
        #F.relu is the function that Max(x,0)
        #for example, input has two dimensions x1, x2
        # in the hidden layer, node 1 = Sigmoid (w1x1+w2x2+c) 
        # here we use relu function instead
        # node 1 = Max(w1x1+w2x2+c, 0)
        out = self.drop(out)
        out = self.fc2(out)
        
        #usually you only apply relu function to the hidden layer 
        return self.sig(out)