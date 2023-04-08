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
