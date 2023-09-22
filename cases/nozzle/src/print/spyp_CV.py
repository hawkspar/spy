from sys import path

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from handlers import *
from helpers import findStuff

Ss_ref=[1]
ms_ref=[-2]
Sts_ref=np.linspace(0,1,54)
elmts_nb =[222,345,445]

for S in Ss_ref:
	fig, ax1, ax2 = figCreator()
	lines=[]
	for nb in elmts_nb:
		dir="resolvent_"+str(nb)+"k/gains/txt/"
		for m in ms_ref:
			gains=[]
			for St in Sts_ref:
				d,name=findStuff(dir,{'S':S,'m':m,'St':St},distributed=False,return_distance=True)
				if d>1e-2: print("MEF ! Found file distant from reference")
				gains.append(np.max(np.loadtxt(name)))
			lines=plot(ax1,ax2,str(nb)+'k elements',Sts_ref,gains,m,lines,alpha=nb/elmts_nb[-1])
	tinkerFig(ax1,ax2,fig,f"S={S}.png",lines,"./")