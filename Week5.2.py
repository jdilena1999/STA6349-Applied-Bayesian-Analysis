# pip install pystan
import stan  # note: PyStan 3.x import is `stan`, not `pystan`
import matplotlib.pyplot as plt
import numpy as np
from distributions import gamma_pdf
stan_code = """
data {
  int<lower=0> N;
  array[N] int<lower=0> Y;
}
parameters {
  real<lower=0> pi;
}
model {
  Y ~ poisson(pi);
  pi ~ gamma(20, 5);
}
"""
actual_iterations = 5_000
chains = 4
data = {"N":3,
        "Y": (0,1,0)}

posterior = stan.build(stan_code, data=data, random_seed=84735)
fit = posterior.sample(num_chains=chains, num_samples=actual_iterations, num_warmup=actual_iterations)

pi_draws = fit["pi"].reshape((chains,actual_iterations))  # numpy array, shape (1, num_samples * num_chains) since pi is scalar
pi_draws_flat = pi_draws.reshape(1,-1)

chains = {
    
}
colors = ["#B0C4DE","#A8B8C8","#C0CCD9","#7B92A8"]
# Caterpillar
for i,c,color in zip(np.arange(1,5),pi_draws,colors):
    chains.setdefault(str(i),c)
    plt.plot(np.arange(1,actual_iterations+1),c,label=i,color=color)
plt.legend()
plt.show()

# Histogram
for i,c,color in zip(np.arange(1,5),pi_draws,colors):
  plt.hist(pi_draws_flat[0],70,density=True,color=color,label=i)
  plt.legend()
  plt.xticks(np.arange(0,7,0.5))
  plt.show()
  
  
  

xs = np.linspace(0,7,5000)
ys = np.array([gamma_pdf(x,20,1/5) for x in xs])
plt.plot(xs,ys)
plt.legend()
plt.title("Known Posterior Model of lambda")
plt.show()


        
        
        

    

