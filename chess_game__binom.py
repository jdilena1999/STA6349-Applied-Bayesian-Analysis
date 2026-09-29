from distributions import binomial_pdf,dbinom
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np


pis = [0.2,0.5,0.8]
p_pis = [.1,.25,0.65]


'''for pi in pis:
    #d_binom = dbinom(6,6,pi)
    plt.bar(np.arange(0,7),[binomial_pdf(x,6,pi) for x in np.arange(0,7)])
    plt.title(f"pi = {pi}")
    plt.show()'''


sum_p_y1 = 0
for pi,p in zip(pis,p_pis):
    d_binom = dbinom(1,6,pi)
    print(f"P(pi = {pi}): {round(p,4)*100}%")
    print(f"P(y=1 | pi = {pi}) or L(pi = {pi} | y = 1): {round(d_binom,4)*100}%")
    print(f"P(pi = {pi} | y = 1): {round(d_binom * p,4)*100}%")
    print("- - "*50)
    sum_p_y1+=(d_binom*p)

print(f"Probability of y=1 over all possible values of pi: {round(sum_p_y1,4)*100}%")

   
    


    
    