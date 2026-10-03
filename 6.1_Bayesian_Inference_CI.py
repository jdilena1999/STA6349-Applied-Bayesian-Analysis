import pandas as pd
import numpy as np 
from beta_binom import get_posterior_table_beta_binom
from gamma_poisson import get_posterior_table_gamma_poisson
from distributions import pbeta,qbeta,qgamma
data = pd.read_csv("moma_sample.csv")
#print(data.columns)
#print(data[["genx"]].value_counts())

#prior_posterior_df = get_posterior_table_beta_binom(4,6,14,100,plot=True)

#print(prior_posterior_df)

#print(qbeta((.025,0.975),18,92))


data = pd.read_excel("./STA6349-Applied-Bayesian-Analysis/data.xlsx")
#get_posterior_table_gamma_poisson(5,1,data['incidents'].to_numpy(),5,plot=True)
models = get_posterior_table_gamma_poisson(5,1,data['incidents'].to_numpy(),5,plot=True)
print(qgamma((0.025,0.975),148,1/37))



