import pandas as pd
from distributions import pbeta,beta_pdf,binomial_pdf,pbinom,dbinom
import numpy as np
import matplotlib.pyplot as plt
from expectations import beta_expectation,beta_variance,x
x_var = x
# Beta-Binomial
    # Beta Distribution as Prior
        #pi ~ {0,1}
    # Binomial Distribution as additional data
    # Beta Distribution as Posterior also, but with different parameters


# pi ~ Beta(a,B), with a mean or E[pi] of 0.45


def update_prior(prior_alpha,prior_beta,new_y,new_n,df):
    new_alpha = prior_alpha+new_y
    new_beta = prior_beta+new_n-new_y
    new_mean = beta_expectation(x_var,new_alpha,new_beta)
    df['alpha'].append(new_alpha)
    df['beta'].append(new_beta)
    df['model'].append(f'posterior {len(df['model'])}')
    df['variance'].append(beta_variance(new_alpha,new_beta))
    df['mean'].append(new_mean)
    df['probability'].append(dbinom(new_y,new_n,new_mean))
    df['mode'].append(calc_mode(new_alpha,new_beta))
    return (new_alpha,new_beta)

def calc_mode(a,b):
    try:
        return (a-1)/(a+b-2)
    except ZeroDivisionError:
        return np.nan


# Mariokart example:

results2 = {
    'alpha':[1],
    'beta':[1],
    'model':[' Order 1 prior'],
    'mean':[beta_expectation(x_var,1,1)],
    'variance':[beta_variance(1,1)],
    'probability':[''],
    'mode':[calc_mode(1,1)]
}
print("Order 1")
print(update_prior(results2['alpha'][-1],results2['beta'][-1],6,28,results2))
print(update_prior(results2['alpha'][-1],results2['beta'][-1],8,27,results2))
print(update_prior(results2['alpha'][-1],results2['beta'][-1],12,29,results2))
print(update_prior(results2['alpha'][-1],results2['beta'][-1],5,30,results2))
print(pd.DataFrame(results2))





results3 = {
    'alpha':[1],
    'beta':[1],
    'model':['Order 2 prior'],
    'mean':[beta_expectation(x_var,1,1)],
    'variance':[beta_variance(1,1)],
    'probability':[''],
    'mode':[calc_mode(1,1)]
}


print("Order 2")
print(update_prior(results3['alpha'][-1],results3['beta'][-1],5,30,results3))
print(update_prior(results3['alpha'][-1],results3['beta'][-1],12,29,results3))
print(update_prior(results3['alpha'][-1],results3['beta'][-1],6,28,results3))
print(update_prior(results3['alpha'][-1],results3['beta'][-1],8,27,results3))
print(pd.DataFrame(results3))




