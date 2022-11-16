import re, os
import numpy as np
from matplotlib import pyplot as plt

color_code={'-1':'black','0':'blue','1':'red','2':'orange','3':'violet','4':'green','5':'cyan'}

dat={}
dir="/home/shared/cases/nozzle/resolvent/gains/"
file_names = [f for f in os.listdir(dir) if f[-3:]=="txt"]
for file_name in file_names:
	match = re.search(r'm=(-?\d*\.?\d*)', file_name)
	m=match.group(1)
	match = re.search(r'St=(\d*\.?\d*)',file_name)
	St=match.group(1)
	match = re.search(r'Re=(\d*\.?\d*)',file_name)
	Re=match.group(1)
	match = re.search(r'S=(\d*\.?\d*)',file_name)
	S=match.group(1)
	if not Re in dat.keys(): 		dat[Re]      ={}
	if not S  in dat[Re].keys(): 	dat[Re][S]   ={}
	if not m  in dat[Re][S].keys(): dat[Re][S][m]={}
	dat[Re][S][m][St]=np.max(np.loadtxt(dir+file_name))

plt.rcParams.update({'font.size': 26})
for Re in dat.keys():
	for S in dat[Re].keys():

		plt.figure(figsize=(10,10),dpi=200)
		for m in dat[Re][S].keys():
			Sts,gains=[],[]
			for St in dat[Re][S][m].keys():
				Sts.append(float(St))
				gains.append(dat[Re][S][m][St])
			Sts,gains=np.array(Sts),np.array(gains)
			ids=np.argsort(Sts)
			plt.plot(Sts[ids],gains[ids]**2,label=r'$m='+f'{int(m):d}$',color=color_code[str(m)],linewidth=3)
		plt.xlabel(r'$St$')
		plt.ylabel(r'$\sigma^{(1)2}$')
		plt.yscale('log')
		plt.xticks([0,.5,1])
		plt.savefig(dir+f"Re={Re}_S={S}.png")
		plt.close()
		"""
		for m in dat[Re][S].keys():
			plt.plot(Sts[ids],gains[ids]**2,label=r'$m='+f'{int(m):d}$',color=color_code[str(m)])

		plt.xlabel(r'$St$')
		plt.ylabel(r'$\sigma^{(1)2}$')
		plt.yscale('log')
		plt.xticks([0,.5,1])
		plt.ylim(min(dat),max(dat))
		#plt.legend()
		plt.savefig(dir+f"scaled_Re={Re}_S={S}.png")
		plt.close()"""