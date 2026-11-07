import pandas as pd
from glm import stan_glm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import bambi as bmb
import arviz as az

data = pd.read_csv("big_word_club.csv")[["special_ed_observed","score_pct_change","school_id","treat"]].dropna()
data_special_ed = data.groupby("special_ed_observed").agg(count=pd.NamedAgg(column="score_pct_change",aggfunc="count"),mean=pd.NamedAgg(column="score_pct_change",aggfunc="mean"),sd=pd.NamedAgg(column="score_pct_change",aggfunc=lambda x:np.std(x))).reset_index()
print(data_special_ed)

#sns.kdeplot(data=data,x="score_pct_change",hue="special_ed_observed")
#plt.show()
'''sns.kdeplot(data=data,x="score_pct_change")
percentiles = np.quantile(data[["score_pct_change"]].dropna().to_numpy().ravel(),[0.25,0.5,0.75])
std = data[["score_pct_change"]].dropna().to_numpy().ravel().std()
sds = std * np.array([(-3,3),(-2,2),(-1,1)])
mean=data[["score_pct_change"]].dropna().to_numpy().ravel().mean()
print(std)
print(sds)
print(percentiles)

for n,(l,u),c in zip(["3 SD","2 SD","1 SD"],sds,["Red",'Yellow','Green']):
    plt.axvline(l+mean,label=n,color=c,linestyle='--')
    plt.axvline(u+mean,label=n,color=c,linestyle='--')
plt.title(f"Mean: {round(mean,4)}\nSD: {round(std,4)}")
#plt.legend()
plt.show()'''

'''priors_1 = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=50),
    "1|school_id": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=0.1)),
    "sigma": bmb.Prior("Exponential", lam=0.1)
}
model,idata = stan_glm("score_pct_change ~ 1 + (1 | school_id)",data=data,priors=priors_1,tidy=True,plot=True)
'''
'''priors2 = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=50),
    "1|school_id": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=0.1)),
    "special_ed_observed":bmb.Prior("Normal",mu=0,sigma=1),
    "sigma": bmb.Prior("Exponential", lam=0.1)
}
model,idata = stan_glm("score_pct_change ~ special_ed_observed + (1 | school_id)",data=data,priors=priors2,tidy=True,plot=True)
'''

priors3 = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=50),
    "1|school_id": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=0.1)),
    "special_ed_observed":bmb.Prior("Normal",mu=0,sigma=1),
    "treat": bmb.Prior("Normal", mu=0, sigma=10),
    "sigma": bmb.Prior("Exponential", lam=0.1)
}
model,idata = stan_glm("score_pct_change ~ treat + special_ed_observed + (1 | school_id)",data=data,priors=priors3,tidy=True,plot=True)
