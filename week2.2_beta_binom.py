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


def update_prior(prior_alpha,prior_beta,new_num_successes,new_n,df):
    new_alpha = prior_alpha+new_num_successes
    new_beta = prior_beta+new_n-new_num_successes
    new_mean = beta_expectation(x_var,new_alpha,new_beta)
    df['alpha'].append(new_alpha)
    df['beta'].append(new_beta)
    df['model'].append(f'posterior {len(df['model'])}')
    df['variance'].append(beta_variance(new_alpha,new_beta))
    df['sd'].append(np.sqrt(float(beta_variance(new_alpha,new_beta))))
    df['mean'].append(new_mean)
    df['probability'].append(dbinom(new_num_successes,new_n,new_mean))
    df['mode'].append(calc_mode(new_alpha,new_beta))
    return (new_alpha,new_beta)

def calc_mode(a,b):
    try:
        return (a-1)/(a+b-2)
    except ZeroDivisionError:
        return np.nan


# Mariokart example:

def get_posterior_table_beta_binom(prior_a,prior_b,new_num_successes,new_n,results = {},**kwargs):
    if len(results.keys()) == 0:
        results = {
            'alpha':[prior_a],
            'beta':[prior_b],
            'model':['Order 1 prior'],
            'mean':[beta_expectation(x_var,prior_a,prior_b)],
            'variance':[beta_variance(prior_a,prior_b)],
            'sd':[np.sqrt(float(beta_variance(prior_a,prior_b)))],
            'probability':[''],
            'mode':[calc_mode(prior_a,prior_b)]
        }
    update_prior(prior_a,prior_b,new_num_successes,new_n,results)
    print(pd.DataFrame(results))
    prior = [beta_pdf(x,prior_a,prior_b) for x in np.linspace(0,new_n,1001)]
    data = [binomial_pdf(x,new_n,new_num_successes/new_n) for x in np.linspace(0,new_n,1001)]
    posterior = [beta_pdf(x,results['alpha'][-1],results['beta'][-1]) for x in np.linspace(0,new_n,1001)]
    if kwargs.get("plot"):
            plt.plot(np.linspace(0,new_n,1001),prior,label="Prior",color='yellow')
            plt.plot(np.linspace(0,new_n,1001),data,label='data',color='blue')
            plt.plot(np.linspace(0,new_n,1001),posterior,label='Posterior',color='green')
            plt.legend()
            plt.show()
    return results

#get_posterior_table_beta_binom(45,55,30,50,plot=True)



'''results3 = {
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
print(pd.DataFrame(results3))'''




