import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glm import stan_glm
import bambi as bmb
import arviz as az
from plotting_functions import get_line_coordinates


data = pd.read_csv("big_word_club.csv")
data["school_id"] = data["school_id"].astype(str)
print(data.columns)
#print(data.head(10))
data_filtered = data[["school_id","score_pct_change"]][data["treat"]==1].dropna().reset_index()
#print(f"Schools: {', '.join([str(x) for x in list(set(data['school_id'].to_numpy()))])}")
print(f"Number of schools: {len(set(data['school_id'].to_numpy()))}")
students_per_school = {
    "School":[],
    "Number of Students":[]
}
priors = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=50),
    "1|school_id": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=0.1)),
    "sigma": bmb.Prior("Exponential", lam=0.1)
}
schools = sorted(list(set(data["school_id"].to_numpy())))
for s in schools:
    students_per_school["School"].append(s)
    students_per_school["Number of Students"].append(len(set(data["participant_id"][data["school_id"].isin([s])].dropna().to_numpy())))
print(pd.DataFrame(students_per_school))
print(f"Variance between schools: {np.var(students_per_school['Number of Students'])}")
#print(data)
model,idata = stan_glm("score_pct_change ~ 1 + (1|school_id)",data_filtered,priors=priors,plot=True,tidy=True)

'''priors = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=2.5,auto_scale=True),
    "age": bmb.Prior("Normal", mu=0, sigma=2.5,auto_scale=True),
    "sigma": bmb.Prior("Exponential", lam=1,auto_scale=True),
    }
running = pd.read_csv("cherry_blossom_sample.csv")[["runner","age","net"]].dropna()'''
#print(running[running.isna().any(axis=1)])
'''sns.scatterplot(data=running,x="age",y="net")
plt.show()'''
#model = stan_glm("net ~ age",data=running,priors=priors,conf_level=0.8)
'''sns.scatterplot(data=running[running['runner'].astype(str).isin(['1'])],x='age',y='net',hue='runner')
sns.lineplot(data=get_line_coordinates(running,'age',75.2242,0.2678),x="x",y="y")
plt.show()

sns.scatterplot(data=running[running['runner'].astype(str).isin(['20'])],x='age',y='net',hue='runner')
sns.lineplot(data=get_line_coordinates(running,'age',75.2242,0.2678),x="x",y="y")
plt.show()

sns.scatterplot(data=running[running['runner'].astype(str).isin(['22'])],x='age',y='net',hue='runner')
sns.lineplot(data=get_line_coordinates(running,'age',75.2242,0.2678),x="x",y="y")
plt.show()'''
