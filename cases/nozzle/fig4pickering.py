from matplotlib import pyplot as plt
import numpy as np

n=100
#n=20
ms=np.arange(0,6)
Sts=np.linspace(0,1,n)
gains=np.empty(n)

for i,m in enumerate(ms):
	for j,St in enumerate(Sts):
		try: gains[j]=np.loadtxt(f"resolvent/gains_S=0.000_m={m:00.2f}_St={St:00.3f}.dat")
		except OSError: pass
	plt.plot(Sts,gains**2,label=r'$m='+f'{m:00.0f}'+'$')

plt.xlabel(r'$St$')
plt.ylabel(r'$\sigma^{(1)2}$')
plt.yscale('log')
plt.legend()
plt.savefig("fig4.png")