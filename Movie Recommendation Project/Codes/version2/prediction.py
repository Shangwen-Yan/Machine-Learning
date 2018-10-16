import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np

from getdata import getdata
       
X_org, X,num_F = getdata()

def rcmd(X_ts,X_ts_norm):
    #Load models
    autoencoder = torch.load('autoencoder.pkl')
    kmeans = joblib.load("kmeans.m")
    
    
    #find watched movie in movie list for users
    rate = X_ts.numpy()[:,:,0]
 
    #encoder
    encoded, _ = autoencoder(Variable(X_ts_norm.view(-1, 9125*3)))
    hf = encoded.data.numpy()
    #kmeans
    user_label = kmeans.predict(hf)
    #movie list
    movies = np.loadtxt("mvrcmdlist.txt")
    rcmdlist = []
    for u in range(len(X_ts)):
        rcmd = movies[user_label[u]-1].copy()
        uwatched = list(np.where(rate[u,:]!=0))
        for mv in uwatched:
            if mv in rcmd: rcmd.remove(mv)
        rcmdlist.append(rcmd)
        
    #recommend first 10 movies for each user
    return np.array(rcmdlist,dtype=int)[:,:10]

X_ts = X_org[:3].clone()
X_ts_norm = X[:3].clone()
rcmdlist = rcmd(X_ts,X_ts_norm)
for i in range(len(rcmdlist)):
    print('===========================================')
    print('===========================================')
    print('For user ',i+1)
    for j in range(len(rcmdlist[0])):
        print('Movie name: ',movie[rcmdlist[i,j],1],' ||genres: ',movie[rcmdlist[i,j],2])