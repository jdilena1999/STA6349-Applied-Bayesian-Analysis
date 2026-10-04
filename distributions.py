import pandas as pd
import numpy as np
import scipy.stats as stats
from math import e
import matplotlib.pyplot as plt
import sympy as sp
from sympy import factorial


#   Overall Pattern:
#       I. Discrete RVs
#           a. Get probability distribution function (class notes)
#           b. Probability Functions by getting probability of each value from 0 to n or 0 to lambda
#       II. Continuous RVs
#           a. Get probability distribution function (pdf) (class notes)
#           b. Probability Functions by integrating over the pdf from the min to max (-oo, oo or explicit min,max or 0,1)





# =======================
# Discrete Random Variables
# =======================


def binomial_pdf(y,n,p):
    one = factorial(n)/(factorial(y)*factorial(n-y))
    two = p**y
    three = (1-p)**(n-y)
    return one*two*three

def dbinom(x,size,prob):
    xs = np.arange(0,size+1) # => x-values from 0 to n
    ys = [binomial_pdf(x,size,prob) for x in xs] # => probabilities of each x-value
    for a,b in zip(xs,ys):
        if a == x:
            return b # => returns the probability for the specific value passed through the "x" parameter in the function

def pbinom(q,size,prob,lower_tail=True):
    xs = np.arange(0,size+1) # => x-values from 0 to n
    ys = [binomial_pdf(x,size,prob) for x in xs] # => probabilities of each x-value
    final = [] # => initialize final probability list
    for a,b in zip(xs,ys):
        if not lower_tail: # => if lower_tail == False, append probabilities whose respective x values are greater than the value passed as the "q" parameter
            if a > q:
                final.append(b)
        else: # => if lower_tail == True, append probabilities whose respective x values are less than or equal to the value passed as the "q" parameter
            if a <= q:
                final.append(b)

    return np.sum(final) # => return summed probabilities


def poisson_pdf(y,lam):
    one = (lam**y)/factorial(y)
    two = np.exp(-lam)
    return one*two


def dpoisson(x,lam):
    xs = np.arange(0,lam+1)
    ys = [poisson_pdf(x,lam) for x in xs]
    for a,b in zip(xs,ys):
        if a == x:
            return b
    #print(xs)
    #print(ys)


    
def ppoisson(q,lam,lower_tail = True):
    xs = np.arange(0,lam+1)
    ys = [poisson_pdf(x,lam) for x in xs]
    final = []
    for a,b in zip(xs,ys):
        if a <= q:
            final.append(b)
    #print(xs)
    #print(ys)
    if lower_tail:
        return np.sum(final)
    else:
        return 1 - np.sum(final)

                


def uniform_pdf(min,max):
    return 1/(max-min)

def normal_pdf(y,mean,std):
    one = 1/(std*np.sqrt(2*np.pi))
    two = sp.exp(-((y-mean)**2)/(2*std**2))
    return one*two

def gamma_fn(a):
    y = sp.symbols('y')
    f = y**(a-1)*sp.exp(-y)
    integ = sp.integrate(f,(y,0,sp.oo))
    return integ
def gamma_pdf(y,alpha,beta):
    num = (y**(alpha-1))*(sp.exp(-y/beta))
    denom = (beta**alpha) * gamma_fn(alpha)
    return num/denom


    
def punif_cont(q,minimum,maximum,lower_tail=True):
    x = sp.symbols('x')
    f = uniform_pdf(minimum,maximum)
    if lower_tail:
        result = sp.integrate(f,(x,minimum,q))
    else:
        result = sp.integrate(f,(x,q,maximum))
    return result

def pnorm(q,mean,std,minimum=-sp.oo,maximum=sp.oo,lower_tail=True):
    x = sp.symbols('x')
    f = normal_pdf(x,mean,std)
    if lower_tail:
        integ = sp.integrate(f,(x,minimum,q))
    else:
        integ = sp.integrate(f,(x,q,maximum))
    return integ.evalf()

def pgamma(q,shape,scale,lower_tail = True):
    x = sp.symbols('x')
    f = gamma_pdf(x,shape,scale)
    if lower_tail:
        integ = sp.integrate(f,(x,0,q))
    else:
        integ = sp.integrate(f,(x,q,sp.oo))
    return integ.evalf()



def beta(a,b):
    return gamma_fn(a)*gamma_fn(b)/gamma_fn(a+b)

def beta_pdf(y,a,b):
    num = (y**(a-1))*((1-y)**(b-1))
    denom = beta(a,b)
    return num/denom
def pbeta(q,shape1,shape2,lower_tail = True):
    x = sp.symbols('x')
    f = beta_pdf(x,shape1,shape2)
    if lower_tail:
        integ = sp.integrate(f,(x,0,q))
    else:
        integ = sp.integrate(f,(x,q,1))
    return integ

def qbeta_auto(ps:tuple,shape1,shape2):
    return stats.beta.ppf(ps,shape1,shape2)
'''def qbeta_manual(ps:tuple,shape1,shape2):
    xs = np.linspace(0,1,1_000)
    ys = [beta_pdf(x,shape1,shape2) for x in xs]
    final = []
    for p in ps:
        deltas = [abs(p-x) for x in xs]
        target_index = deltas.index(min(deltas))
        final.append(ys[target_index]) 
    return final'''           

def qgamma_auto(ps:tuple,alpha,beta):
    return stats.gamma.ppf(ps,alpha,scale=beta)
'''def qgamma_manual(ps,alpha,beta):
    xs = np.linspace(0,alpha*3)
    ys = [gamma_pdf(x,alpha,beta) for x in xs]    
    final = [] 
    for p in ps:
        deltas = [abs(p-y) for y in ys]
        target_index = np.array(deltas).argmin()
        target_y = ys[target_index]      
        final.append(target_y)
    return final'''
def beta_cdf_auto(x,shape1,shape2):
    return stats.beta.cdf(x,shape1,shape2)
def beta_cdf_manual(rv,shape1,shape2):
    x = sp.symbols('x')
    f = beta_pdf(x,shape1,shape2)
    return sp.integrate(f,(x,0,rv))
    #plt.plot(xs,ys)
    #plt.show()



if __name__ == '__main__':

    
    print(qgamma_auto((0.025,0.975),148,1/37))
    print(qgamma_manual((0.025,0.975),148,1/37))


    '''plt.plot([beta_pdf(x,10,2) for x in np.linspace(0,1,1_000)])
    plt.show()'''
    '''print()
    print("= "*50)
    print("Binomial")
    print("= "*50)

    print(f"P[X = 2]: {dbinom(2,4,0.5)}")
    print(f"P[X > 2]: {pbinom(2,4,0.5,False)}")
    print(f"P[X < 4] aka P[X <= 3]: {pbinom(3,4,0.5)}")

    print()
    print("= "*50)
    print("Poisson")
    print("= "*50)

    print(f"P[No more than 3 customers arrive]: {ppoisson(3,7)}")
    print(f"P[At least 2 customers arrive]: {ppoisson(1,7,False)}")
    print(f"P[Exactly 5 customers arrive]: {dpoisson(5,7)}")


    # =======================
    # Continuous Random Variables
    # =======================


            
    print()
    print("= "*50)
    print("Uniform")
    print("= "*50)

    print(f"P[A worker takes fewer than 13 minutes]: {punif_cont(13,9,15)}")
    print(f"P[A worker takes at least 11 minutes]: {punif_cont(11,9,15,False)}")
    print(f"P[A worker takes between 14 and 15 minutes]: {punif_cont(15,9,15) - punif_cont(14,9,15)}")



    print()
    print("= "*50)
    print("Normal")
    print("= "*50)

    print(f"P[Carrot between 10 and 13 cm]: {pnorm(13,11.5,1.15) - pnorm(10,11.5,1.15)}")
    print(f"P[Carrot less than 9 cm]: {pnorm(9,11.5,1.15)}")
    print(f"P[Carrot 12 cm or larger]: {pnorm(12,11.5,1.15,lower_tail=False)}")

    print()
    print("= "*50)
    print("Gamma")
    print("= "*50)


    print(f"Proportion of incomes greater than $100,000: {pgamma(100_000,32,2500,False)}")
    print(f"Proportion of incomes between $75,000 and $150,000: {pgamma(150_000,32,2500) - pgamma(75_000,32,2500)}")


    print()
    print("= "*50)
    print("Beta")
    print("= "*50)

    print(f"P[Fewer than 60% of respondents like the new flavor]: {pbeta(0.6,8,2)}")
    print(f"P[More than 90% of respondents like the new flavor]: {pbeta(0.9,8,2,False)}")
    print(f"P[Between 70% and 90% of respondents like the new flavor]: {pbeta(0.9,8,2) - pbeta(0.7,8,2)}")

    '''




