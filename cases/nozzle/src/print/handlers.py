import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

color_code={-5:'lightgreen',-4:'darkgreen',-3:'cyan',-2:'tab:blue',-1:'darkblue',0:'black',1:'darkred',2:'tab:red',3:'darkorange',4:'magenta',5:'tab:pink'}
marker_code={0:None,1:4,2:3,3:2,4:1}
CV_code={345:'+',445:'x'}

square=False
sig_lbl_pre=r'$\sigma^{('
sig_lbl_app=')'+'2'*square+'}$'
x_lims=[0,2]
y_lims=[20,70000]
lims_zoom=[0,.05]
plt.rcParams.update({'font.size': 26})

def figCreator():
	fig = plt.figure(figsize=(20,10),dpi=500)
	gs = fig.add_gridspec(1, 2, wspace=0)
	ax1, ax2 = gs.subplots(sharey=True)
	return fig, ax1, ax2

def plot(ax1,ax2,lbl,Sts,gains,m,lines,ax_zoom=None,nb=None,scatter=False):
	# Transform into nice arrays
	ids=np.argsort(Sts)
	Sts,gains=(np.array(Sts)*2)[ids],(np.array(gains)**(1+square))[ids]
	if scatter:
		if m>=0: lines.append(ax2.scatter(Sts,gains,label=lbl,marker=CV_code[nb],color=color_code[ m]))
		if m<=0:			  ax1.scatter(Sts,gains,		  marker=CV_code[nb],color=color_code[-m])
	else:
		gains_spl = CubicSpline(Sts, gains)
		Sts_fine=np.linspace(x_lims[0],x_lims[1],1001)
		# Plotting
		if m>=0: lines.append(ax2.plot(Sts_fine,gains_spl(Sts_fine),label=lbl,color=color_code[ m],marker=str(marker_code[ m]),markevery=[500],markersize=12,linewidth=3)[0])
		if m<=0:			  ax1.plot(Sts_fine,gains_spl(Sts_fine),		  color=color_code[-m],marker=str(marker_code[-m]),markevery=[500],markersize=12,linewidth=3)
		if not ax_zoom is None: ax_zoom.plot(Sts_fine,gains_spl(Sts_fine),label=r'$m='+str(m)+'$',color=color_code[m],marker=str(marker_code[m]),markevery=[(60*m-450)*(m>0)+800],markersize=12,linewidth=3)
	return lines

def tinkerFig(ax1,ax2,fig,name,lines,plt_dir):
	# Plot gains on a common figure
	fig.subplots_adjust(right=.85)
	fig.legend(handles=lines,bbox_to_anchor=(1., .65))
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