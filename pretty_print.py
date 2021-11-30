import numpy as np
from matplotlib import pyplot as plt

vals_real,vals_imag=np.loadtxt("validation/eigenvalues/evals_S=1.000_m=0_sigma=-0.130+0.970j.dat",unpack=True)
# Plot them all!
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(vals_imag,vals_real,edgecolors='k',facecolors='none')
#plt.plot([-1e1,1e1],[0,0],'k--')
ax.set_aspect(1)
#plt.axis([-3,3,-.15,.15])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\sigma$')
plt.savefig("validation/eigenvalues_S=1.000_m=0.svg")