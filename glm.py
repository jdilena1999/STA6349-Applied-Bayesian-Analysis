import bambi as bmb
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def tidy(idata, var_names, conf_level=0.95):
    lo, hi = (1 - conf_level) / 2, 1 - (1 - conf_level) / 2
    rows = []
    for v in var_names:
        da = idata.posterior[v]
        extra_dims = [d for d in da.dims if d not in ("chain", "draw")]
        if not extra_dims:
            items = [(v, da.stack(sample=("chain", "draw")).values)]
        else:
            stacked = da.stack(sample=("chain", "draw"))
            dim = extra_dims[0]
            items = [(f"{v}[{lab}]", stacked.sel({dim: lab}).values)
                     for lab in stacked[dim].values]
        for name, d in items:
            med = np.median(d)
            rows.append({
                "term": name,
                "estimate": med,
                "std.error": 1.4826 * np.median(np.abs(d - med)),
                "conf.low": np.quantile(d, lo),
                "conf.high": np.quantile(d, hi),
            })
    return pd.DataFrame(rows)
    '''lo, hi = (1 - conf_level) / 2, 1 - (1 - conf_level) / 2
    rows = []
    for v in var_names:
        d = idata.posterior[v].stack(sample=("chain", "draw")).values
        med = np.median(d)
        rows.append({
            "term": v,
            "estimate": med,
            "std.error": 1.4826 * np.median(np.abs(d - med)),   # MAD SD
            "conf.low": np.quantile(d, lo),
            "conf.high": np.quantile(d, hi),
        })
    
    return pd.DataFrame(rows)'''

def stan_glm(y_modeled_over_x:str,data:pd.DataFrame,priors:dict,family="gaussian",chains=4,iter=5_000*2,seed=84735,conf_level = 0.95,**kwargs) -> tuple:
    model = bmb.Model(y_modeled_over_x,data=data,family=family,priors=priors)
    idata = model.fit(draws=int(iter/2),tune=int(iter/2),random_seed=seed)
    estimate_df = tidy(idata,list(priors.keys()),conf_level)
    v_names = [x for x in list(priors.keys())]
    if kwargs.get("tidy"):
        print(estimate_df)
    if kwargs.get("plot"):
        print(az.summary(idata, var_names=v_names))
        az.plot_trace(idata, var_names=v_names)
        az.plot_forest(idata, var_names=[x for x in v_names if x not in ("Intercept","sigma")], combined=True, r_hat=True)
        '''print(az.summary(idata, var_names=v_names))
        az.plot_trace(idata, var_names=v_names)
        az.plot_forest(idata, var_names=v_names-["Intercept","sigma"], combined=True, r_hat=True)'''
        plt.show()
    return model,idata
if __name__ == "__main__":
    df = pd.read_csv("bikes.csv")
    #print(df.head())
    fig,ax = plt.subplots()
    sns.scatterplot(data=df,x="temp_feel",y="rides")
    #plt.show()
    priors = {
        "Intercept": bmb.Prior("Normal", mu=5_000, sigma=1_000,auto_scale=True),
        "temp_feel": bmb.Prior("Normal", mu=100, sigma=40),
        "sigma": bmb.Prior("Exponential", lam=0.0008),
    }
    model = bmb.Model("rides ~ temp_feel", data=df, family="gaussian",priors=priors,dropna=True)
    idata = model.fit(draws=5_000, tune=5_000, chains=4, random_seed=84735)
    model.predict(idata,kind='response',data=df)
    estimate_df = tidy(idata, ["Intercept", "temp_feel", "sigma"])
    print(estimate_df)
    pp_matrix = (
        idata.posterior_predictive["rides"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "__obs__")
        .values
    )
    predicted_data_y_values = np.linspace(40,100,1000) * estimate_df["estimate"][estimate_df["term"]=="temp_feel"].values + estimate_df["estimate"][estimate_df["term"]=="Intercept"].values
    predicted_data_y_values_low = np.linspace(40,100,1000) * estimate_df["conf.low"][estimate_df["term"]=="temp_feel"].values + estimate_df["conf.low"][estimate_df["term"]=="Intercept"].values
    predicted_data_y_values_high = np.linspace(40,100,1000) * estimate_df["conf.high"][estimate_df["term"]=="temp_feel"].values + estimate_df["conf.high"][estimate_df["term"]=="Intercept"].values
    sns.lineplot(data={'x':np.linspace(40,100,1000),'y':predicted_data_y_values},x='x',y='y',linestyle='--',c='orange',label="estimate")
    sns.lineplot(data={'x':np.linspace(40,100,1000),'y':predicted_data_y_values_low},x='x',y='y',linestyle='--',c='red',label="95 CI")
    sns.lineplot(data={'x':np.linspace(40,100,1000),'y':predicted_data_y_values_high},x='x',y='y',linestyle='--',c='red',label="95 CI")

    plt.show()
#az.summary(idata)