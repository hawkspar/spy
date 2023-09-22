from sys import path
from re import search
from os import listdir
from matplotlib import pyplot as plt

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from handlers import *
from helpers import dirCreator, findStuff

dir="/home/shared/cases/nozzle/resolvent/gains/"
dat_dir,plt_dir=dir+"txt/",dir+"plots/"
dirCreator(dir)
dirCreator(dat_dir)
dirCreator(plt_dir)
stick_to_ref=True # Use all available gains or limit to those specified in setup ?
Ss_ref=[1]#np.linspace(0,1,6)
ms_ref=[2,-2]#range(-4,5)
zoom,suboptimals,CV=False,False,True

Sts_CV=np.linspace(0,1,26)
elmts_nb =[345,445]

# Read all the gains in a dictionary
dat={}
for file_name in listdir(dat_dir):
	if file_name[-3:]=="txt":
		match = search(r'Re=(\d*)',file_name)
		Re=int(match.group(1))
		match = search(r'S=(\d*\,?\d*)',file_name)
		S=float(match.group(1).replace(',','.'))
		match = search(r'm=(-?\d*)', file_name)
		m=int(match.group(1))
		match = search(r'St=(\d*\,?\d*e?(\+|-)?\d*)',file_name)
		St=float(match.group(1).replace(',','.'))
		if not Re in dat.keys(): 		dat[Re]      ={}
		if not S  in dat[Re].keys():	dat[Re][S]   ={}
		if not m  in dat[Re][S].keys(): dat[Re][S][m]={}
		dat[Re][S][m][St] = np.loadtxt(dat_dir+file_name).reshape(-1)

for Re in dat.keys():
	for S in dat[Re].keys():
		if stick_to_ref and (not np.any(np.isclose(S,Ss_ref,atol=1e-3))): continue
		# Loop on data, extract and reorder properly
		ms=list(dat[Re][S].keys())
		ms.sort()
		fig, ax1, ax2 = figCreator()
		lines=[]
		if zoom:
			fig_zoom = plt.figure(figsize=(13,10),dpi=500)
			ax_zoom = plt.subplot(111)
		else: ax_zoom=None
		for m in ms: # Pretty ms in order
			if stick_to_ref and (not np.any(np.isclose(m,ms_ref,atol=.5))): continue
			if suboptimals:
				n=3
				Sts,gains=[[] for _ in range(n+1)],[[] for _ in range(n+1)]
				fig, ax1, ax2 = figCreator()
				lines=[]
			else: Sts,gains=[],[]
			for St in dat[Re][S][m].keys():
				#if stick_to_ref and (not np.any(np.isclose(float(St),Sts_ref,atol=.005))): continue
				if suboptimals:
					for i in range(min(dat[Re][S][m][St].size,n+1)):
						Sts[i].append(St)
						gains[i].append(dat[Re][S][m][St][i])
				else:
					Sts.append(St)
					gains.append(np.max(dat[Re][S][m][St]))
			if suboptimals:
				for i in range(n): lines=plot(ax1,ax2,r'$i='+str(i)+'$',Sts[i],gains[i],m,lines,alpha=1-.7*i/(n-1))
				tinkerFig(ax1,ax2,fig,f"suboptimals_S={S}_m={m}.png",lines,plt_dir=plt_dir)
			else: lines=plot(ax1,ax2,r'$|m|='+str(m)+'$',Sts,gains,m,lines,ax_zoom)
			if CV:
				for nb in elmts_nb:
					dir_CV="resolvent_"+str(nb)+"k/gains/txt/"
					gains=[]
					for St in Sts_CV:
						d,name=findStuff(dir_CV,{'S':S,'m':m,'St':St},distributed=False,return_distance=True)
						if d>1e-2: print("MEF ! Found file distant from reference")
						gains.append(np.max(np.loadtxt(name)))
					lines=plot(ax1,ax2,str(nb)+'k',Sts_CV,gains,m,lines,nb=nb,scatter=True)
		tinkerFig(ax1,ax2,fig,f"S={S}"+CV*"_CV"+".png",lines,plt_dir)

		if zoom:
			# Plot a log-log zoom around origin
			ax_zoom.set_xlabel(r'$St$')
			ax_zoom.set_ylabel(sig_lbl_pre+'1'+sig_lbl_app)
			ax_zoom.set_yscale('log')
			ax_zoom.set_xlim(lims_zoom)
			plt.grid('x')
			box = ax_zoom.get_position()
			ax_zoom.set_position([box.x0, box.y0, box.width*10/13, box.height])
			ax_zoom.legend(loc='center left',bbox_to_anchor=(1, 0.5))
			fig_zoom.savefig(plt_dir+f"S={S}_zoom.png")
			plt.close(fig_zoom)