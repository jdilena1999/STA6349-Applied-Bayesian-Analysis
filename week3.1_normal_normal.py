
import pandas as pd
from distributions import normal_pdf,pnorm
import numpy as np
import matplotlib.pyplot as plt
from expectations import x,normal_variance,normal_expectation
from math import factorial

# Normal-Normal Model
## mean is between 6 & 7
## sample of n = 25
## for now, assume std = 0.5

prior_mu = 6.5
prior_std = 0.4
plausible_range = (prior_mu-2*prior_std,prior_mu+2*prior_std)
print(plausible_range)
#plt.plot(np.linspace(5,8,1001),[normal_pdf(x,prior_mu,prior_std) for x in np.linspace(5,8,1001)],label='Prior',color='yellow')
#plt.grid(visible=True)
#plt.show()
x_var = x
'''for s,r in zip([5],[10]):
    xs = np.linspace(0,25,1001)
    ys = [gamma_pdf(a,s,1/r) for a in xs]
    plt.plot(xs,ys)
    #plt.ylim(top=1)
    plt.title(f"shape: {s} | rate: {r}")
    plt.grid(visible=True)
    plt.show()'''
    

    
def update_prior(prior_mu,prior_std,data_std,y_bar,n,df):
    prior_var = prior_std**2
    data_var = data_std**2
    new_mu_1 = prior_mu*(data_var/(n*prior_var+data_var))
    new_mu_2 = y_bar*(n*prior_var/(n*prior_var+data_var))
    new_mu = new_mu_1+new_mu_2
    new_sd = (prior_var*data_var)/(n*prior_var+data_var)

    #new_mean = gamma_expectation(x_var,new_alpha,new_beta)
    df['mu'].append(new_mu)
    df['sd'].append(np.sqrt(new_sd))
    df['var'].append(new_sd)
    df['model'].append(f"posterior {len(df['model'])}")
    #df['variance'].append(normal_variance(new_mu,new_sd))
    #df['mean'].append(normal_expectation(x_var,new_mu,new_sd))
    #df['probability'].append(poisson_joint_probability(ys,lam))
    return (new_mu,new_sd)


def get_posterior_table_normal_normal(prior_mean,prior_sd,new_data,results = {},**kwargs):
    if len(results.keys()) == 0:
        results = {
            'mu':[prior_mean],
            'var':[prior_sd**2],
            'sd':[prior_sd],
            'model':['prior'],
            #'mean':[normal_expectation(x_var,6.5,0.4)],
            #'variance':[normal_variance(6.5,0.4)],
            #'probability':[0]
        }
    #data = pd.read_csv("football.csv")
    data_mu = np.mean(new_data)
    data_std = np.std(new_data)#data[data['group'].isin(["fb_concuss"])]['volume'].std()
    data_n = len(new_data)#data[data['group'].isin(["fb_concuss"])]['volume'].to_numpy())

    posterior_new = update_prior(prior_mu,prior_std,data_std,data_mu,data_n,results)
    print(pd.DataFrame(results))
    #print(posterior_new)
    prior = [normal_pdf(x,results['mu'][0],results['sd'][0]) for x in np.linspace(5,8,1001)]
    posterior = [normal_pdf(x,results['mu'][1],results['sd'][1]) for x in np.linspace(5,8,1001)]
    if kwargs.get("plot"):
        plt.plot(np.linspace(5,8,1001),prior,label="Prior",color='yellow')
        plt.hist(new_data,label='data',color='blue',density=True,bins=30)
        plt.plot(np.linspace(5,8,1001),posterior,label='Posterior',color='green')
        plt.legend()
        plt.show()
#data = pd.read_csv("football.csv")
#data = data[data["group"].isin(["fb_concuss"])]['volume'].to_numpy()
#get_posterior_table_normal_normal(6.5,0.4,data,plot=False)

#plt.plot(data[data['group'].isin(["fb_concuss"])]['volume'],[normal_pdf(x,data[data['group'].isin(["fb_concuss"])]['volume'].mean(),data[data['group'].isin(["fb_concuss"])]['volume'].std()) for x in data[data['group'].isin(["fb_concuss"])]['volume']],label='Data',color='blue')
#plt.legend()
#plt.show()