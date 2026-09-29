import pandas as pd
import numpy as np
import sympy as sp
from distributions import normal_pdf,beta_pdf,gamma_pdf,poisson_pdf,uniform_pdf,binomial_pdf
x = sp.symbols('x')
def normal_expectation(expectation,mean,std):
    #x = sp.symbols("x")
    integ = sp.integrate(expectation*normal_pdf(x,mean,std),(x,-sp.oo,sp.oo))
    return integ.evalf()
def normal_variance(mean,std):
    e_x = normal_expectation(x,mean,std)
    return normal_expectation((x-e_x)**2,mean,std)
def poisson_expectation(expectation,lam):
    #lam = sp.nsimplify(lam)
    try:
        integ = sp.Sum(expectation*poisson_pdf(x,lam),(x,0,sp.oo))
    except:
        integ = sp.Sum(expectation*poisson_pdf(x,sp.nsimplify(lam)),(x,0,sp.oo))
    return integ.doit().evalf()
def poisson_variance(lam):
    e_x = sp.simplify(poisson_expectation(x,lam))
    return sp.simplify(poisson_expectation((x-e_x)**2,lam))
def binomial_expectation(expectation,n,p):
    n = sp.Integer(n)
    p = sp.nsimplify(p)
    sum = sp.Sum(expectation*binomial_pdf(x,n,p),(x,0,n))
    return sum.doit().evalf()
def binomial_variance(n,p):
    e_x = sp.simplify(binomial_expectation(x, n, p))
    e_x2 = sp.simplify(binomial_expectation(x**2, n, p))
    return sp.simplify(e_x2 - e_x**2)
def beta_expectation(expectation,a,b):
    a = sp.Integer(a)
    b = sp.Integer(b)
    #p = sp.nsimplify(p)
    sum = sp.integrate(expectation*beta_pdf(x,a,b),(x,0,1))
    return sum.evalf()
def beta_variance(a,b):
    e_x = sp.simplify(beta_expectation(x, a,b))
    e_x2 = sp.simplify(beta_expectation(x**2, a,b))
    return sp.simplify(e_x2 - e_x**2)
def gamma_expectation(expectation,a,b):
    a = sp.Float(a)
    b = sp.Float(b)
    final = sp.integrate(expectation*gamma_pdf(x,a,b),(x,0,sp.oo))
    return final.doit().evalf()
def gamma_variance(a,b):
    e_x = sp.simplify(gamma_expectation(x, a,b))
    e_x2 = sp.simplify(gamma_expectation(x**2, a,b))
    return sp.simplify(e_x2 - e_x**2)



if __name__ == '__main__':
    print(gamma_expectation(x,10,2))
    print(gamma_variance(10,2))