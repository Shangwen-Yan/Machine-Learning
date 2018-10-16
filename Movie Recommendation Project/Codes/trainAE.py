# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:32:32 2018

@author: chenz
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

from getdata import getdata
       
_, X = getdata()

###autoencoder
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(9125*3, 2281*3),
            nn.Tanh(),
            nn.Linear(2281*3, 570*3),
            nn.Tanh(),
            nn.Linear(570*3, 142*3),
            nn.Tanh(),
            nn.Linear(142*3, 36*3),
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(36*3, 142*3),
            nn.Tanh(),
            nn.Linear(142*3, 570*3),
            nn.Tanh(),
            nn.Linear(570*3, 2281*3),
            nn.Tanh(),
            nn.Linear(2281*3, 9125*3),
            nn.Sigmoid(), #0-1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()

###Training
LR = 0.0001
EPOCH = 10

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

#save training loss in this file
f = open('loss.txt', 'w')

for epoch in range(EPOCH):
    X_tr = Variable(X.view(-1, 9125*3))
    
    encoded, decoded = autoencoder(X_tr)

    loss = loss_func(decoded, X_tr)      # mean square error
    f.write(str(loss.data[0])+' ')
    print('epoch:',epoch,', loss:', loss.data[0])
    optimizer.zero_grad()               # clear gradients for this training step
    loss.backward()                     # backpropagation, compute gradients
    optimizer.step()                    # apply gradients   

f.close()    

print('Saving this model')
torch.save(autoencoder, 'autoencoder.pkl')