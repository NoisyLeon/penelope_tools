import numpy as np
import numba
import matplotlib.pyplot as plt

def h(l, y=50., b=100., r=5.):
    return y*np.log(b*np.exp(-l)+r) - b*np.exp(-l)+r

def dhdl(l, y=50., b=100., r=5.):
    return (1-y/(b*np.exp(-l)+r))*b*np.exp(-l) 





l=np.arange(50)*0.1

h1  = h(l=l)
plt.plot(l,h1, 'b-', lw=3, label='h(l)')

li=1
c   = -2.*(h(l=0) - h(li) + dhdl(l=li)*li)/li**2
h2  = (h(l=li)) + (dhdl(l=li))*(np.arange(50)*0.1 - li) - c/2.*(l-li)**2

plt.plot(l,h2, 'k--', lw=3, label='h(li), li=1')
li=2
c   = -2.*(h(l=0) - h(li) + dhdl(l=li)*li)/li**2
h2  = (h(l=li)) + (dhdl(l=li))*(np.arange(50)*0.1 - li) - c/2.*(l-li)**2
plt.plot(l,h2, 'b--', lw=3, label='h(li), li=2')
li=3
c   = -2.*(h(l=0) - h(li) + dhdl(l=li)*li)/li**2
h2  = (h(l=li)) + (dhdl(l=li))*(np.arange(50)*0.1 - li) - c/2.*(l-li)**2
plt.plot(l,h2, 'g--', lw=3, label='h(li), li=3')
li=4
c   = -2.*(h(l=0) - h(li) + dhdl(l=li)*li)/li**2
h2  = (h(l=li)) + (dhdl(l=li))*(np.arange(50)*0.1 - li) - c/2.*(l-li)**2
plt.plot(l,h2, 'r--', lw=3, label='h(li), li=4')

plt.ylim(50, 160)
plt.legend(fontsize=20)
plt.show()