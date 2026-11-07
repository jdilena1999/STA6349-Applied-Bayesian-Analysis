import pandas as pd
import numpy as np
from glm import stan_glm
import matplotlib.pyplot as plt
import seaborn as sns
import bambi as bmb
import arviz as az


data = pd.read_csv("spotify.csv")[['artist','title','popularity']]

# Complete Pooled
'''print(data)
print(data.shape)
print(len(set(data['artist'].to_numpy())))'''
artist_means = data.groupby("artist").agg(num_titles=pd.NamedAgg(column='title',aggfunc='count'),mean_popularity=pd.NamedAgg(column="popularity",aggfunc='mean')).reset_index()
print(artist_means.sort_values(by="num_titles",ascending=True))
#sns.kdeplot(data=data,x="popularity",fill=True)
#plt.show()
'''model,idata = stan_glm("popularity ~ 1",data,{
    "Intercept":bmb.Prior("Normal", mu=50, sigma=2.5,auto_scale=True),
    "sigma":bmb.Prior("Exponential",lam=1,auto_scale=True)},family="gaussian",chains=4,iter=5_000*2
    ,conf_level=0.8,tidy=True)'''


# No Pooled

'''sns.kdeplot(data=data,x="popularity",hue="artist")
plt.show()'''
model,idata = stan_glm("popularity ~ artist - 1",data,{
    "artist":bmb.Prior("Normal", mu=50, sigma=2.5,auto_scale=True),
    "sigma":bmb.Prior("Exponential",lam=1,auto_scale=True)},family="gaussian",chains=4,iter=5_000*2
    ,conf_level=0.8,tidy=True,plot=True)
