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
