# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:40:55 2018

@author: chenz

@return :
    X_org 
    X
    num_F
    X_test：把Y_test中相关电影的rating制0（假装没看过）。格式和X一样
    Y_test:test users实际看过的电影（取了timstamp最近的10个）
"""
import numpy as np
import pandas as pd 
import torch
import torchvision


def getdata_t():
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
    
    
    ### get index of movieId
    dict_m = {1:0}
    def getIndex(movieId):
        index = dict_m.get(movieId)
        if(index!=None):
            return int(index)
        index = np.where(movie[:,0] == movieId)[0]
        if(index >= 0 ):
            dict_m[movieId]=index
            return int(index);
        else:
            return -1;
        
    ##select 20 users as testing
    n_user = users.shape[0]
    n_movie = movie.shape[0]
    n_user_test = 20;
    n_user_train = n_user - n_user_test
    n_f = 3
    X = torch.zeros(n_user_train,n_movie, n_f) # | rate ########| mean_rate | genre(int) |
    X_test = torch.zeros(n_user_test,n_movie, n_f) # | rate ########| mean_rate | genre(int) |
    Y_test = {} #{userId:[[movieId,rating],[2,2.2],[3,3.3]],2:[[2,1.1],[9,2.2],[3,3.3]]}
    print(X.shape)
    print(X_test.shape)
    train_index = (rating[:,0] <= n_user_train)
    test_index = (rating[:,0] > n_user_train)
    rating_train = rating[train_index,:]
    rating_test_ori = rating[test_index,:]
    rating_test = np.zeros((1,4)) - 1
    print('train shape of rating: ',rating_train.shape)
    print('test shape of rating: ',rating_test.shape)
    
    
    df = pd.DataFrame(rating_test_ori, columns=['userId','movieId','rating','timestamp'])
    grouped = df.groupby(df['userId'])
    for key,group in grouped:
        userId = key
        data = group.sort_values(['timestamp'],ascending=False).as_matrix()
        Y_test[userId] = data[:15,1:3]
        #print(rating_test.shape)
        #print(data[10:,:].shape)
        rating_test = np.vstack((rating_test,data[15:,:]))


    ###Form data with shape(users, movies, features)

    #mean_rate
    matrix_rate = np.zeros((n_movie,2))
    print(matrix_rate.shape)
    print(rating_train.shape) #(20000263, 4)
    for i in range(rating_train.shape[0]):
        userId,movieId,rate = rating_train[i,0:3]
        movieId = int(movieId)
        movieIndex = getIndex(movieId)
        if(movieIndex <= n_movie and movieIndex != -1 and rate != str and userId <= n_user_train):
            matrix_rate[movieIndex,0] = matrix_rate[movieIndex,0] + rate
            matrix_rate[movieIndex,1] = matrix_rate[movieIndex,1] + 1
        if(i%1000000 == 0):
            print('Conting mean rates /(ㄒoㄒ)/~~')
    zero_index = (matrix_rate[:,1] == 0)
    matrix_rate[zero_index] = 1
    mean_rate = matrix_rate[:,0]/matrix_rate[:,1]


    #save user's rate
    for i in range(rating_train.shape[0]):
        userId,movieId,rate = rating[i,0:3]
        movieId =int( movieId)
        movieIndex = getIndex(movieId)
        if(movieIndex <= n_movie and movieIndex != -1 and rate != str): #有的rating里的movieId,MOVIE表里没有
            X[int(userId - 1),int(movieIndex),int(0)] = rate
        if(i%1000000 == 0):
            print('Saving users rates /(ㄒoㄒ)/~~')
            
    for i in range(rating_test.shape[0]):
        userId,movieId,rate = rating_test[i,0:3]
        movieId =int( movieId)
        movieIndex = getIndex(movieId)
        if(movieIndex <= n_movie and movieIndex != -1 and rate != str):   #有的rating里的movieId,MOVIE表里没有
            X_test[int(userId -n_user_train- 1),int(movieIndex),0] = rate
        if(i%1000000 == 0):
            print('Saving users rates /(ㄒoㄒ)/~~')

    #save mean_rate
    for i in range(n_movie):
        X[:,i,1] = mean_rate[i]
        X_test[:,i,1] = mean_rate[i]
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
        movieIndex = getIndex(movieId)
        if(movieIndex <= n_movie and movieIndex != -1 and movieIndex != -1):
            X[:,movieIndex,2] = float(x)
            X_test[:,movieIndex,2] = float(x)
    
    ###normalize data
    X_org = X.clone()
    Xmean = torch.mean(X, 1, True)
    Xstd = torch.var(X, 1, True)
    zero_index1 = (Xstd == 0)
    Xstd[zero_index1] = 1 
    torchvision.transforms.Normalize(Xmean, Xstd)(X)
    Xmean = torch.mean(X_test, 1, True)
    Xstd = torch.var(X_test, 1, True)
    zero_index1 = (Xstd == 0)
    Xstd[zero_index1] = 1 
    torchvision.transforms.Normalize(Xmean, Xstd)(X_test)
    
    print('Loading pregress secceeded!!! *★,°*:.☆(￣▽￣)/$:*.°★* 。')
    print('X.size:',X.size())
    print('X_test.size:',X_test.size())

    return X_org, X,81,X_test,Y_test