
import pandas as pd
from distributions import gamma_pdf,pgamma,poisson_pdf,ppoisson
import numpy as np
import matplotlib.pyplot as plt
from expectations import x,gamma_variance,gamma_expectation
from math import factorial

# Gamma-Poisson Model
## Most likely value of lambda is 5, but ranges from 2 to 7
## This Means E[lambda] = s/r = 5...meaning r = s/5 and s = 5r
## FYI - beta = 1/r
x_var = x
'''for s,r in zip([5],[1]):
    xs = np.linspace(0,25,1001)
    ys = [gamma_pdf(a,s,1/r) for a in xs]
    plt.plot(xs,ys)
    #plt.ylim(top=1)
    plt.title(f"shape: {s} | rate: {r}")
    plt.grid(visible=True)
    plt.show()'''
    
# We pick s = 1 and r = 1

# Find the joint pmf by multiplying y values across all of our days the P(y | lambda=5)
# Updates original paramaters by:
    # s+= sum of all ys,
    # r+=n

#data = pd.read_excel("./Week_6/data.xlsx")['incidents'].to_numpy()      # New Data
#data = [6,2,2,1]

def poisson_joint_probability(ys,lam):
    final_list = []
    for y in ys:
        num = (lam**y)*(np.e**-lam)
        denom = factorial(y)
        final_list.append(num/denom)
    start = 1
    for l in final_list:
        start = start*l
    return start

    
def update_prior(prior_s,prior_r,new_ys,df,lam):
    sum_y = np.sum(new_ys)
    y_mult = 1
    for a in new_ys:
        y_mult = y_mult * a
    #avg = np.mean(ys)
    new_n = len(new_ys)
    new_s = prior_s+sum_y
    new_r = prior_r+new_n
    #new_mean = gamma_expectation(x_var,new_alpha,new_beta)
    df['s'].append(new_s)
    df['r'].append(new_r)
    df['model'].append(f"posterior {len(df['model'])}")
    df['variance'].append(gamma_variance(new_s,1/new_r))
    df['SD'].append(np.sqrt(float(gamma_variance(new_s,1/new_r))))
    df['mean'].append(gamma_expectation(x_var,new_s,1/new_r))
    df['probability'].append(poisson_joint_probability(new_ys,lam))
    return (new_s,new_r)



def get_posterior_table_gamma_poisson(prior_s,prior_r,new_data,lam,results = {},**kwargs):
    if len(results.keys()) == 0:
        results = {
            's':[prior_s],
            'r':[prior_r],
            'model':['prior'],
            'mean':[gamma_expectation(x_var,prior_s,1/prior_r)],
            'variance':[gamma_variance(prior_s,1/prior_r)],
            'SD':[np.sqrt(float(gamma_variance(prior_s,1/prior_r)))],
            'probability':['']
        }
    posterior_new = update_prior(prior_s,prior_r,new_data,results,lam)
    #print(posterior_new)
    prior = [gamma_pdf(x,prior_s,1/prior_r) for x in np.linspace(0,11,1001)]
    data = [poisson_pdf(x,lam) for x in np.linspace(0,11,1001)]
    posterior = [gamma_pdf(x,results['s'][-1],1/results['r'][-1]) for x in np.linspace(0,11,1001)]
    print(pd.DataFrame(results))
    
    if kwargs.get("plot"):
        plt.plot(np.linspace(0,11,1001),prior,label="Prior",color='yellow')
        plt.plot(np.linspace(0,11,1001),data,label='data',color='blue')
        plt.plot(np.linspace(0,11,1001),posterior,label='Posterior',color='green')
        plt.legend()
        plt.show()
    return results
#get_posterior_table_gamma_poisson(15,3,[6,2,2,1],5,plot=True)



#print(np.mean(data))

#print(ppoisson(3,3.89189,False))
