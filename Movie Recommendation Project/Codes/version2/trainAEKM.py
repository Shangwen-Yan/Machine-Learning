# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:32:32 2018

@author: chenz
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np

from getdata import getdata
       
_, X,num_F = getdata()
num_In = X.size()[1]*X.size()[2]

###autoencoder##########################################################################
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_In, num_F*2),
            nn.Tanh(),
            nn.Linear(num_F*2, num_F),
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(num_F, num_F*2),
            nn.Tanh(),
            nn.Linear(num_F*2, num_In),
            #nn.Sigmoid(), #0-1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()

###Training
LR = 1
EPOCH = 1000

AELoss = []
optimizer = torch.optim.SGD(autoencoder.parameters(), weight_decay=0.001, lr=LR,momentum=0.9)
loss_func = nn.MSELoss()

#save training loss in this file
f = open('loss.txt', 'w')

for epoch in range(EPOCH):
    X_tr = Variable(X.view(-1, num_In))
    
    encoded, decoded = autoencoder(X_tr)

    loss = loss_func(decoded, X_tr)      # mean square error
    AELoss.append(loss.data[0])
    f.write(str(loss.data[0])+' ')
    if epoch%10==0:
        print('epoch:',epoch,', loss:', loss.data[0])
    optimizer.zero_grad()               # clear gradients for this training step
    loss.backward()                     # backpropagation, compute gradients
    optimizer.step()                    # apply gradients   

f.close()    
plt.plot(range(len(AELoss)),AELoss)
plt.title('training loss')
plt.savefig("trloss.png")
print('Done training, Saving this model')
torch.save(autoencoder, 'autoencoder.pkl')

###kmeans##########################################################################
print('clusteringo(*￣▽￣*)ブ')
encoded, _ = autoencoder(X_tr)
hf = encoded.data.numpy()

n_clusters = 9 
#Since there is 18 genres for movies, 18 or 9 clusters for users migtht be reasonable
kmeans = KMeans(n_clusters=n_clusters).fit(hf)

joblib.dump(kmeans, "kmeans.m")
User_label = kmeans.labels_
User_mean = kmeans.cluster_centers_

###Store movies for each cluster###################################################
print('storing recommond list for users')
movies = []
X_n = X.numpy()
for i in range(n_clusters):
    usersi = X_n[np.array(User_label)==i,:,0]### get all users in i cluster, features: rate
    SuperUseri = np.sum(usersi,0) ###sum up all user's rating for i cluster
    mvid = np.argsort(-SuperUseri,0)
    movies.append(mvid)
np.savetxt("mvrcmdlist.txt",movies)