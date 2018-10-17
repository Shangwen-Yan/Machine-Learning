--- Based on Encoder and K-Means

====================================

Zhaoxi Chen | N18963553 | zc1134

Shangwen Yan | N17091204 | sy2160

## Abstract
recommender systems have become increasingly popular in recent years, and are utilized in a variety of areas including movies, music, news, books, research articles, search queries, social tags, and products in general. 
Here we implement a movie recommendation system based on Auto-Encoder and K-Meanse. The dataset is the popular one : movie-lens.

## Introduction
With the increasing development of technology and Internet, more and more customers choose to spend their spare time online instead of off-line. As a result, it's becoming of great importance to make a good use of the data produced by the consumers to offer a better consumption environment which is of great benefit both for merchants and customers.
As one of the results, recommender systems have become increasingly popular in recent years, and are utilized in a variety of areas including movies, music, news, books, research articles, search queries, social tags, and products in general. Here we implement a movie recommendation system using movie-lens dataset.

## Related works
### movie recommendation
Traditional way is to take the user, predict the ratings for all the movies from movies' catalog, sort the movies in descending order by the predicted ratings, then take the top 10 movies and recommend them. Some CNN methods also used in movie recommendation system. But all of this algorithm only focus on genres of movies but not user's characteristics.

### Encoder+K-means
This combination actually used more often in image classification. It's main idea is first move redundancy in images by an encoder. And then clustering images with encoded image can speed up the whole algorithm.

## Dataset
### Original dataset
Here we use the open dataset MovieLens. The relevant information is movies’ratings,genres,tags given by users
Number of users:7120，Number of movies: 27278，Number of ratings: 1048575
__Files we use:__
|movie.csv|||
|-|-|-|
|movieId|title| genres|

|rating.csv||||
|-|-|-|-|
|userId|movieId| rating|timestamp|

### Training data
Since it's a clustering problem, it's more important to train the clusters. We only use 7100 users for traing, and make the data in to a torch of size (7100,27278,3) which means (number of users, number of movies, (rating, mean_rating, genres)). Since genres are given as text, we use one-hot-coding to turn it into binary int, and then decimal int.
![image](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/C608ADA4760D219EC2FC89DEDCD83DB3.png)
### Testing data
We only use 20 users for testing case. And for those testing users, we split the most recently 20 ratings used to compared with the recommended movies, and the past records to go through the encoder and k-means to do the recommendation

## Architecture
The whole architecture is: first we use encoder to reduce the number of features which makes it simpler for the kmeans part, then use k-means to group the users into 9 clusters. After calculating the scores movies in each clusters, we recommend 30 movies which have not been seen by the users to them.
![image](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/E0E82EAC9A56BF24EDAE8F58CA3FED98.png)
### Encoder part
![image](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/475E22E4FE0B8816883BA6808A2A92BC.png)

First, we  train an autoencoder to make the decoded and input matrix as samilar as possible.

To make sure AE’s weights  are  orthogonal , we add penalty to __MSE loss:__

$L(W) = MSE +\lambda (W^TW-I)$

__epoch vs loss:__

|__epoch vs loss__|__input vs decoded__|
|-|-|
|![屏幕快照 2018-05-02 下午2.46.45.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/A86D494400FAC494E9AF67DD8D215EBD.png)| ![屏幕快照 2018-05-02 下午2.52.36.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/0240B0E21D96089848A1DCE30FC35253.png)|


The encoded tensor represents input tensor well, but with fewer features, which makes it easier for clustering  in k-means part.


###k-means part:
Then use encoded  tensor to do a k-means clustering which generates 9 clusters of user-groups.
For each cluster(user_group), we sum up users’ ratings for each movie and sort the movies by rating desc. 
Then use the top 50 movies’ genres to generate a word cloud to see which kind of movie is popular in that user group.

|word cloud|![cluster1.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/4821ED74077B951BEE77951DC8DE6C23.png)|![cluster2.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/F53227A097566779106E7534318307FE.png)|![cluster3.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/0B84CDED01FDD11330EAE70C169761E6.png)|
|-|-|-|-|
|__bar plot__|![bar1.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/EDCEA6F5F5B2C821818CAD9B1F9FFC80.png)|![bar2.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/38A0893717D88238CCAB7DCAD3001320.png)| ![bar3.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/3C7F4567D983653BAF916270BBCC6D80.png )|
|__after scaling__|![屏幕快照 2018-05-02 下午4.16.48.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/99E22C7D0163716CCB78F26852168625.png)|![屏幕快照 2018-05-02 下午4.16.56.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/2571C6C359104DFFE36D95029E839C84.png)|![屏幕快照 2018-05-02 下午4.17.05.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/C257DEB132DE9A511CA5DE6B3E443D0E.png)|



|word cloud|![cluster4.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/87DF664CAFFC3CC0E94158E92528A307.png)|![cluster5.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/E59758C364180611F6A2DB41703FAFFC.png)|![cluster6.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/1BDB08844298F51430D1EBB7B9959D96.png)|
|-|-|-|-|
|__bar plot__|![bar4.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/BEF8CFBCD629D198C3F7005E57CE6DC2.png)|![bar5.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/12DAB00A88A366217472153FBD81380D.png)|![bar6.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/25625425DFA52CAE7CCDAB4934EE0CD7.png)|
|__after scaling__|![屏幕快照 2018-05-02 下午4.17.12.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/444D3CA06720E10E6D204EF88D7EF453.png)|![屏幕快照 2018-05-02 下午4.17.20.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/06BDCD8F7757B77EECC9859C54CE7B9A.png)|![屏幕快照 2018-05-02 下午4.17.28.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/07CBFD82EC4316080D3F936C12528D6C.png)|


|word cloud|![cluster7.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/DA7CB38609B24DC5D76F3853A427EDF8.png)|![cluster8.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/799E94B681F2B6E53C5A14FCE22EDA39.png)|![cluster9.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/56333F709ECA35CE3B821B9A8F33C822.png)|
|-|-|-|-|
|__bar plot__|![bar7.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/DC4FB35F9D3B4D1673BA0C7D05CC56AF.png)|![bar8.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/8D03DFE5F7854B771EDD88944059CD68.png)|![bar9.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/BE72256C787CE3D9E970C7AACCB2915D.png) |
|__after scaling__|![屏幕快照 2018-05-02 下午4.17.37.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/8876CA3E834BE8F2949ADB8F558AA1EF.png)|![屏幕快照 2018-05-02 下午4.17.52.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/E088F84CF7D6A7504C849CCA9A9D825B.png)|![屏幕快照 2018-05-02 下午4.18.03.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/A7581E7E967B3C80A27E4F7CFF847D1B.png)|

Since ‘Drama’ and ‘Comedy show up frequently, we do a genre analysis among all movies(no clusters).
![bar.png](resources/B362A1760B89E988419C27499818CCCA.png)
From the plot we could find some genres, like Drama and Comedpngy,  are ‘common types’ which are specified to most movies. And thus, most clusters contains this two genres is reasonable.
So we scale the frequency of genres based on their probabilities among all movies. As we can see, the difference after scaling is kind of obvious.

## Recommen Movies
For every users go through the encoder and kmeans to put him into a cluster. Then recommend those movies with highest scores rated by users in the same cluster to him.
![屏幕快照 2018-05-02 上午1.27.32.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/2C876F589BF21E090F48A4C2A510F7A2.png)
## Test Case
Recommend 30 movies to a user based on his past watching history.  Compare the results with those movies he  most recently watched.
here shoes one user's result:
![屏幕快照 2018-05-02 上午1.01.07.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/B12A98C5E1B1B02477BB73F65712CDF8.png)
![屏幕快照 2018-05-02 上午1.01.19.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/706EE04321FEDA5BBE1DB20EC858B839.png)
![屏幕快照 2018-05-02 上午1.01.33.png](https://github.com/Shangwen-Yan/Machine_Leaning/blob/master/Movie%20Recommendation%20Project/resources/76AD975344B616206ED712BEB04B4C30.png)
__Recommend:__
Beauty and the Beast
Speed

__Actually watched:__
Beauty and the Beast
Speed 2



