import pandas as pd
import numpy as np 
from beta_binom import get_posterior_table_beta_binom
from distributions import pbeta,qbeta,qgamma
data = pd.read_csv("moma_sample.csv")
#print(data.columns)
#print(data[["genx"]].value_counts())

#prior_posterior_df = get_posterior_table_beta_binom(4,6,14,100,plot=True)

#print(prior_posterior_df)

#print(qbeta((.025,0.975),18,92))

print(qgamma((0.025,0.975),5,1))



