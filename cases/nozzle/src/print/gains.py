from sys import path
from re import search
from os import listdir
from matplotlib import pyplot as plt

from scipy.optimize import fmin
from scipy.interpolate import CubicSpline, interp1d

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from helpers import dirCreator

color_code={'-5':'lightgreen','-4':'darkgreen','-3':'cyan','-2':'tab:blue','-1':'darkblue','0':'black','1':'darkred','2':'tab:red','3':'darkorange','4':'magenta','5':'tab:pink'}
marker_code={'-4':'1','-3':'2','-2':'3','-1':'4','0':None,'1':'4','2':'3','3':'2','4':'1'}
#color_code={'0':'tab:blue','1':'tab:red','2':'orange','3':'rebeccapurple','4':'olivedrab','5':'cyan'} # Pickering colorscheme
dir="/home/shared/cases/nozzle/resolvent/gains/"
dat_dir,plt_dir=dir+"txt/",dir+"plots/"
dirCreator(dir)
dirCreator(dat_dir)
dirCreator(plt_dir)
stick_to_ref=True # Use all available gains or limit to those specified in setup ?
Ss_ref=np.linspace(0,1,6)
ms_ref=range(-4,5)
square=False
zoom,suboptimals=False,False
sig_lbl_pre=r'$\sigma^{('
sig_lbl_app=')'+'2'*square+'}$'
x_lims=[0,2]
y_lims=[20,70000]
lims_zoom=[0,.05]
# Read all the gains in a dictionary
dat={}
for file_name in listdir(dat_dir):
	if file_name[-3:]=="txt":
		match = search(r'm=(-?\d*)', file_name)
		m=match.group(1)
		match = search(r'St=(\d*\,?\d*e?(\+|-)?\d*)',file_name)
		St=match.group(1).replace(',','.')
		match = search(r'Re=(\d*)',file_name)
		Re=match.group(1)
		match = search(r'S=(\d*\,?\d*)',file_name)
		S=match.group(1)
		if not Re in dat.keys(): 		dat[Re]      ={}
		if not S  in dat[Re].keys():	dat[Re][S]   ={}
		if not m  in dat[Re][S].keys(): dat[Re][S][m]={}
		dat[Re][S][m][St] = np.loadtxt(dat_dir+file_name).reshape(-1)

def figCreator():
	fig = plt.figure(figsize=(20,10),dpi=500)
	gs = fig.add_gridspec(1, 2, wspace=0)
	ax1, ax2 = gs.subplots(sharey=True)
	return fig, ax1, ax2

def plot(ax1,ax2,lbl,Sts,gains,m,lines,ax_zoom=None):
	# Transform into nice arrays
	ids=np.argsort(Sts)
	Sts,gains=(np.array(Sts)*2)[ids],(np.array(gains)**(1+square))[ids]
	gains_spl = CubicSpline(Sts, gains)
	Sts_fine=np.linspace(x_lims[0],x_lims[1],1001)
	# Plotting
	if int(m)>=0: lines.append(ax2.plot(Sts_fine,gains_spl(Sts_fine),label=lbl,color=color_code[m],				  marker=marker_code[m],markevery=[500],markersize=12,linewidth=3)[0])
	if int(m)<=0:			   ax1.plot(Sts_fine,gains_spl(Sts_fine),		   color=color_code[str(abs(int(m)))],marker=marker_code[m],markevery=[500],markersize=12,linewidth=3)
	if not ax_zoom is None: ax_zoom.plot(Sts_fine,gains_spl(Sts_fine),label=r'$m='+m+'$',color=color_code[m],marker=marker_code[m],markevery=[(60*int(m)-450)*(int(m)>0)+800],markersize=12,linewidth=3)
	return lines

def tinkerFig(ax1,ax2,fig,name,lines):
	# Plot gains on a common figure
	fig.subplots_adjust(right=.85)
	fig.legend(handles=lines,bbox_to_anchor=(1, .65))
	ax1.set_title(r'$m<0$')
	ax1.set_xlabel(r'$St$')
	ax1.set_ylabel(sig_lbl_pre+'1'+sig_lbl_app)
	ax1.set_yscale('log')
	ax1.set_xlim(x_lims)
	ax1.set_ylim(y_lims)
	ax1.invert_xaxis()
	ax2.set_title(r'$m>0$')
	ax2.set_xlabel(r'$St$')
	ax2.set_xlim(x_lims)
	ax2.set_ylim(y_lims)
	fig.savefig(plt_dir+name)
	plt.close(fig)

plt.rcParams.update({'font.size': 26})
for Re in dat.keys():
	for S in dat[Re].keys():
		if stick_to_ref and (not np.any(np.isclose(float(S.replace(',','.')),Ss_ref,atol=.05))): continue
		# Loop on data, extract and reorder properly
		ms=[int(m) for m in dat[Re][S].keys()]
		ms.sort()
		ms=[str(m) for m in ms]
		fig, ax1, ax2 = figCreator()
		lines=[]
		if zoom:
			fig_zoom = plt.figure(figsize=(13,10),dpi=500)
			ax_zoom = plt.subplot(111)
		else: ax_zoom=None
		for m in ms: # Pretty ms in order
			if stick_to_ref and (not np.any(np.isclose(int(m),ms_ref,atol=.5))): continue
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
						Sts[i].append(float(St))
						gains[i].append(dat[Re][S][m][St][i])
				else:
					Sts.append(float(St))
					gains.append(np.max(dat[Re][S][m][St]))
			if suboptimals:
				for i in range(n):
					lines=plot(ax1,ax2,r'$i='+str(i)+'$',Sts[i],gains[i],m)
				tinkerFig(ax1,ax2,fig,f"suboptimals_Re={Re}_S={S}_m={m}.png")
			else:
				lines=plot(ax1,ax2,r'$|m|='+m+'$',Sts,gains,m,lines,ax_zoom)
		
		tinkerFig(ax1,ax2,fig,f"Re={Re}_{S}.png")

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
			fig_zoom.savefig(plt_dir+f"Re={int(Re)}_S={S}_zoom.png")
			plt.close(fig_zoom)