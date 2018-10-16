import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.misc import imread

path = 'ml-latest-small'
movie = pd.read_csv("./"+path+"/movie.csv")
movie = np.array(movie)
movies = np.loadtxt("mvrcmdlist1000.txt",dtype=int)
i = 1
back_coloring = imread("cloud.jpg")
for clus in movies:
    m_clus = movie[clus[:50],2]
    wl_space_split = " ".join(m_clus)
    replaced = wl_space_split.replace("|"," ")
    result = ' '.join(replaced.split())
    my_wordcloud = WordCloud(background_color="white",mask=back_coloring).generate(result) 
    plt.figure(i)
    plt.imshow(my_wordcloud)
    plt.axis("off")
    plt.savefig("wrdcnt/cluster%d.png"%(i))
    i +=1