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
