#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
values_a = rand.randint(20, size=20)
values_b = rand.randint(20, size=20)
plt.plot(values_a)
plt.plot(values_b)

#%%
import sys
print(sys.version)