import re, os
import numpy as np
from matplotlib import pyplot as plt

color_code={'0':'blue','1':'red','2':'yellow','3':'violet'}

dat={}
dir="/home/shared/cases/nozzle/resolvent/gains/"
file_names = [f for f in os.listdir(dir) if f[-3:]=="txt"]
for file_name in file_names:
	match = re.search(r'_m=(\d*\.?\d*)', file_name)
	m=match.group(1)
	match = re.search(r'_St=(\d*\.?\d*)',file_name)
	St=match.group(1)
	match = re.search(r'_Re=(\d*\.?\d*)',file_name)
	Re=match.group(1)
	match = re.search(r'_S=(\d*\.?\d*)',file_name)
	S=match.group(1)
	if not Re in dat.keys(): 		dat[Re]      ={}
	if not S  in dat[Re].keys(): 	dat[Re][S]   ={}
	if not m  in dat[Re][S].keys(): dat[Re][S][m]={}
	dat[Re][S][m][St]=np.max(np.loadtxt(dir+file_name))

for Re in dat.keys():
	for S in dat[Re].keys():
		for m in dat[Re][S].keys():
			Sts,gains=[],[]
			for St in dat[Re][S][m].keys():
				Sts.append(float(St))
				gains.append(dat[Re][S][m][St])
			plt.plot(Sts,np.array(gains)**2,label=r'$m='+f'{int(m):d}$',color=color_code[str(m)])

		plt.xlabel(r'$St$')
		plt.ylabel(r'$\sigma^{(1)2}$')
		plt.yscale('log')
		plt.xticks([0,.5,1])
		plt.legend()
		plt.savefig(f"fig4_Re={Re}_S={S}.png")
		plt.close()