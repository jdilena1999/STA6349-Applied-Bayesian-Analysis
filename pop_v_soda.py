import pandas as pd
import numpy as np

data = pd.read_csv("./Week_2/pop_v_soda.csv")

data.replace({'pop':{True:1,False:0}},inplace=True)

data1 = data['pop'].value_counts(normalize=True)
print(data1)

data2 = pd.crosstab(data['pop'],data['region'],normalize='columns')


print(data2)

data3 = data['region'].value_counts(normalize=True)
print(data3)
for x in data2.columns:
    print(f"{x}")
    print(f"P({x}): {data3[x]}")
    print(f"P(Pop | {x}: {data2[x][1]}")
    prob_x_given_pop = data3[x] * data2[x][1] / data1[1]
    print(f"P({x} | Pop): {prob_x_given_pop}")

