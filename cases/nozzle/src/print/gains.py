from re import search
from os import listdir
from matplotlib import pyplot as plt

from setup import *
from helpers import dirCreator

color_code={'-5':'lightgreen','-4':'darkgreen','-3':'cyan','-2':'lightblue','-1':'darkblue','0':'black','1':'darkred','2':'tab:red','3':'darkorange','4':'magenta','5':'tab:pink'}
dir="/home/shared/cases/nozzle/resolvent/gains/"
dirCreator(dir+"plots/")
stick_to_ref=False # Use all available gains or limit to those specified in setup ?
lim_zoom=[1e-12,.2]
# Read all the gains in a dictionary
dat={}
file_names = [f for f in listdir(dir+"txt/") if f[-3:]=="txt"]
for file_name in file_names:
	match = search(r'm=(-?\d*)', file_name)
	m=match.group(1)
	match = search(r'St=(\d*\,?\d*(e-\d\d)?)',file_name)
	St=match.group(1).replace(',','.')
	match = search(r'Re=(\d*)',file_name)
	Re=match.group(1)
	match = search(r'S=(\d*\,?\d*)',file_name)
	S=match.group(1)
	if not Re in dat.keys(): 		dat[Re]      ={}
	if not S  in dat[Re].keys():	dat[Re][S]   ={}
	if not m  in dat[Re][S].keys(): dat[Re][S][m]={}
	dat[Re][S][m][St] = np.loadtxt(dir+"txt/"+file_name).reshape(-1)

plt.rcParams.update({'font.size': 26})
for Re in dat.keys():
	for S in dat[Re].keys():
		if stick_to_ref and not np.any(np.isclose(float(S.replace(',','.')),Ss_ref,atol=.05)): continue
		# Loop on data, extract and reorder properly
		ms=[int(m) for m in dat[Re][S].keys()]
		ms.sort()
		ms=[str(m) for m in ms]
		fig = plt.figure(figsize=(13,10),dpi=200)
		ax = plt.subplot(111)
		fig_zoom = plt.figure(figsize=(13,10),dpi=200)
		ax_zoom = plt.subplot(111)
		for m in ms: # Pretty ms in order
			if stick_to_ref and not np.any(np.isclose(int(m),ms_ref,atol=.5)): continue
			Sts,gains=[],[]
			for St in dat[Re][S][m].keys():
				if stick_to_ref and not np.any(np.isclose(float(St),Sts_ref,atol=.005)): continue
				Sts.append(float(St))
				gains.append(np.max(dat[Re][S][m][St]))
			Sts,gains=np.array(Sts),np.array(gains)
			ids=np.argsort(Sts)
			Sts,gains=Sts[ids],gains[ids]**2 # Usual St=fD/U not fR/U
			ax.plot(Sts,gains,label=r'$m='+m+'$',color=color_code[m],linewidth=3)
			ax_zoom.plot(Sts,gains,label=r'$m='+m+'$',color=color_code[m],linewidth=3)
		# Plot all gains
		ax.set_xlabel(r'$St$')
		ax.set_ylabel(r'$\sigma^{(1)2}$')
		ax.set_yscale('log')
		ax.set_xticks=[0,.5,1,1.5,2]
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width*10/13, box.height])
		ax.legend(loc='center left',bbox_to_anchor=(1, 0.5))
		fig.savefig(dir+"plots/"+f"Re={int(Re)}_S={S}.png") # Usual Re also based on D

		# Plot a log-log zoom around origin
		ax_zoom.set_xlabel(r'$St$')
		ax_zoom.set_ylabel(r'$\sigma^{(1)2}$')
		ax_zoom.set_yscale('log')
		ax_zoom.set_xlim(lim_zoom)
		box = ax_zoom.get_position()
		ax_zoom.set_position([box.x0, box.y0, box.width*10/13, box.height])
		ax_zoom.legend(loc='center left',bbox_to_anchor=(1, 0.5))
		fig_zoom.savefig(dir+"plots/"+f"Re={int(Re)}_S={S}_zoom.png") # Usual Re also based on D

		# Plotting suboptimals
		n=3
		for m in dat[Re][S].keys(): # pretty ms in order
			fig = plt.figure(figsize=(13,10),dpi=200)
			ax = plt.subplot(111)
			Sts,gains=[[] for _ in range(n+1)],[[] for _ in range(n+1)]
			for St in dat[Re][S][m].keys():
				if stick_to_ref and not np.any(np.isclose(float(St),Sts_ref,atol=.01)): continue
				for i in range(min(dat[Re][S][m][St].size,n+1)):
					Sts[i].append(float(St))
					gains[i].append(dat[Re][S][m][St][i])
			for i in range(n):
				Sts[i],gains[i]=np.array(Sts[i]),np.array(gains[i])
				ids=np.argsort(Sts[i])
				ax.plot(Sts[i][ids],gains[i][ids]**2,label=r'$i='+f'{i+1}$',color=color_code[m],alpha=(3*n/2-i)/3/n*2,linewidth=3) # Usual St=fD/U not fR/U
			plt.xlabel(r'$St$')
			plt.ylabel(r'$\sigma^{(1)2}$')
			plt.yscale('log')
			plt.xticks([0,.5,1,1.5,2])
			box = ax.get_position()
			ax.set_position([box.x0, box.y0, box.width*10/13, box.height])
			plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
			plt.savefig(dir+"plots/"+f"Re={int(Re)}_S={S}_m={m}.png")
			plt.close()