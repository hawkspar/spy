import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

n=1000
R=np.flip(2-np.geomspace(1,2,n))
A=np.linspace(1,10,n)
RR,AA=np.meshgrid(R,A)
U=np.tanh(AA*(1-RR**2))
ths = np.trapz(U*(1-U)*RR,R,axis=1)

plt.plot(A,ths)
plt.xlabel(r'$a$')
plt.ylabel(r'$\theta_J$')
plt.savefig("a_to_thetas.png")

ths_f=interp1d(A,ths,bounds_error=False)
ths=np.linspace(.03,.09,5)
for th in ths:
	if th==ths[0]: a=fsolve(lambda a: ths_f(a)-th,5.1)
	else:		   a=fsolve(lambda a: ths_f(a)-th,a)
	print("th=",th,"a=",a[0])