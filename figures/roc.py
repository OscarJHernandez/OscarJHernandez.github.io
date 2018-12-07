import numpy as np
import matplotlib.pyplot as plt

t = 1.0

x = np.arange(-5.0,5.0,.01)
xt = np.arange(t,5.0,0.01)

y0 = np.zeros(len(xt))
y1 = np.exp(-0.5*(x-1)**2)
y11 = np.exp(-0.5*(xt-1)**2)
y2 = np.exp(-0.3*(x+1)**2)
y22= np.exp(-0.3*(xt+1)**2)

plt.plot(x,y1,linewidth=3,color="blue",label="$T_p(x)$")
plt.plot(x,y2,linewidth=3,color="red",label="$T_n(x)$")
plt.fill_between(xt,y11,y22,color="blue",alpha=0.6)
plt.fill_between(xt,y0,y22,color="red",alpha=0.6)
plt.axvline(x=t,linewidth=3,color="green")
plt.legend()
plt.show()
