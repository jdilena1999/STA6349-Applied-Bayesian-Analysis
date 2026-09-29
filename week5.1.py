import pandas as pd 
import distributions as dist 
import expectations as expect 
import numpy as np 
import matplotlib.pyplot as plt


# Beta-Binomial 1
observed_successes = 9
probs_success = np.linspace(0,1,500)
print(probs_success)
final_df = {
    'PI_GRID':[],
    'PRIOR':[],
    'LIKELIHOOD':[],
    'UNNORMALIZED':[],
    #'POSTERIOR':[]

}
for pi in probs_success:
    final_df['PI_GRID'].append(pi)
    final_df['PRIOR'].append(dist.dbeta(pi,2,2))
    final_df['LIKELIHOOD'].append(dist.dbinom(9,10,pi))
    final_df['UNNORMALIZED'].append(dist.dbinom(9,10,pi)*dist.dbeta(pi,2,2))
    #final_df['POSTERIOR'].append((dist.dbinom(9,10,pi)*dist.dbeta(pi,2,2)) / sum(final_df['UNNORMALIZED']))
print(sum(final_df['UNNORMALIZED']))
final_df = pd.DataFrame(final_df)
final_df['POSTERIOR'] = final_df['UNNORMALIZED'] / sum(final_df['UNNORMALIZED'].to_numpy())
print(sum(final_df['POSTERIOR'].to_numpy()))
print(final_df)
#for pi in probs_success:
plt.bar(final_df['PI_GRID'],final_df['POSTERIOR'],width=0.2,color='black',edgecolor='black')

plt.show()