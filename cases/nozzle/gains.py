import re, os
import numpy as np
from matplotlib import pyplot as plt

color_code={'-5':'lightgreen','-4':'darkgreen','-3':'cyan','-2':'lightblue','-1':'darkblue','0':'black','1':'darkred','2':'tab:red','3':'darkorange','4':'magenta','5':'tab:pink'}

dat={}
dir="/home/shared/cases/nozzle/resolvent/gains/"
file_names = [f for f in os.listdir(dir) if f[-3:]=="txt"]
for file_name in file_names:
	match = re.search(r'm=(-?\d*)', file_name)
	m=match.group(1)
	match = re.search(r'St=(\d*\,?\d*)',file_name)
	St=match.group(1).replace(',','.')
	match = re.search(r'Re=(\d*)',file_name)
	Re=match.group(1)
	match = re.search(r'S=(\d*\,?\d*)',file_name)
	S=match.group(1)
	if not Re in dat.keys(): 		dat[Re]      ={}
	if not S  in dat[Re].keys():	dat[Re][S]   ={}
	if not m  in dat[Re][S].keys(): dat[Re][S][m]={}
	dat[Re][S][m][St]=np.max(np.loadtxt(dir+file_name))

plt.rcParams.update({'font.size': 26})
for Re in dat.keys():
	for S in dat[Re].keys():

		fig = plt.figure(figsize=(13,10),dpi=200)
		ax = plt.subplot(111)
		for m in color_code.keys(): # pretty ms in order
			Sts,gains=[],[]
			try:
				for St in dat[Re][S][m].keys():
					if float(St)<.05: continue
					Sts.append(float(St))
					gains.append(dat[Re][S][m][St])
				Sts,gains=np.array(Sts),np.array(gains)
				ids=np.argsort(Sts)
				ax.plot(Sts[ids],gains[ids]**2,label=r'$m='+f'{int(m):d}$',color=color_code[str(m)],linewidth=3)
			except KeyError: pass
		plt.xlabel(r'$St$')
		plt.ylabel(r'$\sigma^{(1)2}$')
		plt.yscale('log')
		plt.xticks([0,.5,1])
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width*10/13, box.height])
		plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
		plt.savefig(f"Re={Re}_S={S}.png")
		plt.close()