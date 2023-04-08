```python
!pip install corner
```
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,uniform
import time
import corner
import scipy.optimize as op
# from numpy import *
```

```python
x,y,yerr = np.loadtxt('https://raw.githubusercontent.com/darshanbeniwal/Astro_data_analysis_w_Python_GHRCE_IUCCA_2023/main/DataFiles/H_30.txt', unpack=True)
```

```python
#Initial seeds
h0_ini,om_ini=60,0.2
h0_ini,om_ini=5,0.9

#Define log-Likelihood Function
def likelihood(theta, x, y, yerr):  
    h0, om= theta
    model = h0* (((om*(1.0+x)**3)+(1.0-om))**0.5)
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(6.28*inv_sigma2)))
```    

```python
#Define Prior Function
def prior(theta):
    h0, om= theta
    if 0.0< h0 < 100.0 and 0.0 < om < 1.0:
        return 0.0
    return -np.inf
    
#Degine Posterior Function
def posterior(theta, x, y, yerr):
    lp = prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return (lp + likelihood(theta, x, y, yerr))
```

```python
#Define number of parameter, steps, burn-in phase
ndim=2
nsteps=10000
nburn_in=100

initials=h0_ini,om_ini
```
```python
start_time = time.time()


#Define Metropolis-Hastings Function
def Metropolis_Hastings(parameter_init, iteration_time):
    result = []
    result.append(parameter_init)
    for t in range(iteration_time):
        step_var = [1, 0.01]
        proposal = np.zeros(2)
        for i in range(2):
            proposal[i] = norm.rvs(loc=result[-1][i], scale=step_var[i])
            probability = np.exp(posterior(proposal,x,y,yerr) - posterior(result[-1],x,y,yerr))
            if (uniform.rvs() < probability):
                result.append(proposal)
            else:
                result.append(result[-1]) 
    return (result)

result = Metropolis_Hastings(initials, nsteps)

result = result[nburn_in:]
a_result = np.zeros(nburn_in)
b_result = np.zeros(nburn_in)
samples_MH=np.array(result)
h0_mcmc, om_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples_MH,
                       [16, 50, 84],axis=0)))

print("----------------------------------------------------------------------\n")
print("Execution time with steps=%s------> %6.3f seconds" % 
      (nsteps,(time.time() - start_time)))
print("----------------------------------------------------------------------\n")
```

```python
# Plot the traceplots of the MCMC chains
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
samples = samples_MH.T

# Plot the traceplot of H0
axes[0].plot(samples[0], "g", alpha=0.8)
axes[0].set_ylabel("$H_0$")

# Plot the traceplot of Om
axes[1].plot(samples[1], "r", alpha=0.8)
axes[1].set_ylabel("$\Omega_{m0}$")

plt.tight_layout()
# plt.savefig("FLCDM_H_31_MH_traces.png")
plt.show()
```

```python
print("""MCMC result:
    H0 = {0[0]} +{0[1]} -{0[2]}
    Om = {2[0]} +{2[1]} -{2[2]}
""".format(h0_mcmc, h0_ini, om_mcmc, om_ini))
```

```python
fig = corner.corner(samples_MH,bins=50,color="b",labels=["$H_0$","$\Omega_{m0}$"],
                    truths=[h0_mcmc[0],om_mcmc[0]],fill_contours=True,
                    levels=(0.68,0.95,0.99,),
                    quantiles=[0.16, 0.5, 0.84],title_fmt='.3f',plot_datapoints=False,smooth=True, 
                    smooth1d=True,show_titles=True)
#fig.savefig("FLCDM_H_31_MH.png")

```

