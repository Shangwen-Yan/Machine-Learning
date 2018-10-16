# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:40:55 2018

@author: chenz
"""
import numpy as np
import pandas as pd 
import torch
import torchvision

def getdata():
    ###Load data
    #path = 'movielens-20m-dataset'
    path = 'ml-latest-small'
    print("===========================================================")
    print("Loading ratings")
    rating = pd.read_csv("./"+path+"/rating.csv")
    print(rating[:3])
    rating = np.array(rating)
    print("===========================================================")
    print("Loading movies")
    movie = pd.read_csv("./"+path+"/movie.csv")
    print(movie[:3])
    movie = np.array(movie)
    print("===========================================================")
    print("Loading tags")
    tag = pd.read_csv("./"+path+"/tag.csv")
    print(tag[:3])
    tag = np.array(tag)
    print("===========================================================")
    print("Loading genome_scores")
    genome_scores = pd.read_csv("./"+path+"/genome_scores.csv")
    print(genome_scores[:3])
    genome_scores = np.array(genome_scores)
    print("===========================================================")
    print("Loading genome_tags")
    genome_tags = pd.read_csv("./"+path+"/genome_tags.csv")
    print(genome_tags[:3])
    genome_tags = np.array(genome_tags)
    print("===========================================================")
    print("Ok (*^_^*)")

    ###get n_user and n_movie
    print("movie.shape",movie.shape)  
    users1 = np.unique(rating[:,0])
    print("users who rated movies:",users1.shape)  
    users2 = np.unique(tag[:,0])
    print("users who taged movies",users2.shape)  
    users = np.unique(np.hstack((users1,users2)))
    print("number of users:",users.shape) 
    
    n_user = users.shape[0]
    n_movie = movie.shape[0]

    ###Form data with shape(users, movies, features)
    n_f = 3
    X = torch.Tensor(n_user,n_movie, n_f) # | rate ########| mean_rate | genre(int) |

    matrix_rate = np.ones((rating.shape[0],2))
    print(matrix_rate.shape)
    #mean_rate
    print(rating.shape) #(20000263, 4)
    for i in range(rating.shape[0]):
        movieId,rate = rating[i,1:3]
        movieId = movieId.astype(int)
        if(movieId <= n_movie and rate != str):
            matrix_rate[movieId-1,0] = matrix_rate[movieId-1,0] + rate
            matrix_rate[movieId-1,1] = matrix_rate[movieId-1,1] + 1
        if(i%1000000 == 0):
            print('Conting mean rates /(ㄒoㄒ)/~~')
    mean_rate = matrix_rate[:,0]/matrix_rate[:,1]
    #save user's rate
    for i in range(rating.shape[0]):
        userId,movieId,rate = rating[i,0:3]
        movieId = movieId.astype(int)
        if(movieId <= n_movie and rate != str):   #有的rating里的movieId,MOVIE表里没有
            X[userId - 1,movieId - 1,0] = rate
        if(i%1000000 == 0):
            print('Saving users rates /(ㄒoㄒ)/~~')
    #save mean_rate
    for i in range(n_movie):
        X[:,i,1] = mean_rate[i]
        if(i%1000000 == 0):
            print('Saving mean rates /(ㄒoㄒ)/~~')
    genre_list = np.array(['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'])
    #save genre
    for i in range(movie.shape[0]):
        movieId, genres = movie[i,(0,2)]
        genres_split = genres.split('|')
        for genre in genres_split:
            x = 0
            index = np.where(genre_list ==genre)[0]
            if(index.size > 0):
                x = x + 2**index
        if(movieId <= n_movie ):
            X[:,movieId - 1,2] = float(x)

    ###normalize data
    X_org = X.clone()
    Xmean = torch.mean(X, 1, True)
    Xstd = torch.var(X, 1, True)
    torchvision.transforms.Normalize(Xmean, Xstd)(X)
    
    print('Loading pregress secceeded!!! *★,°*:.☆(￣▽￣)/$:*.°★* 。')
    return X_org, X