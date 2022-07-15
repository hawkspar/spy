from matplotlib import pyplot as plt
import numpy as np
import re, os

dat={}
dir="/home/shared/cases/nozzle/resolvent/"
file_names = [f for f in os.listdir(dir) if f[-3:]=="txt"]
for file_name in file_names:
	match = re.search(r'_m=(\d*\.?\d*)', file_name)
	m=match.group(1)
	match = re.search(r'_St=(\d*\.?\d*)',file_name)
	St=match.group(1)
	if not m in dat.keys(): dat[m]={}
	dat[m][St]=np.max(np.loadtxt(dir+file_name))

for m in dat.keys():
	Sts,gains=[],[]
	for St in dat[m].keys():
		Sts.append(float(St))
		gains.append(dat[m][St])
	if m=='1.00': print(Sts)
	plt.plot(Sts,np.array(gains)**2,label=r'$m='+f'{float(m):00.0f}$')

plt.xlabel(r'$St$')
plt.ylabel(r'$\sigma^{(1)2}$')
plt.yscale('log')
plt.xticks([0,.5,1])
plt.legend()
plt.savefig("fig4.png")