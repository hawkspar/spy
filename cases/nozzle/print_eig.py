# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import warnings
import numpy as np
from setup import *
from re import search
from os import listdir
from spy import dirCreator
from matplotlib import pyplot as plt

# Shorthands
color_code={'-5':'lightgreen','-4':'darkgreen','-3':'cyan','-2':'lightblue','-1':'darkblue','0':'black','1':'darkred','2':'tab:red','3':'darkorange','4':'magenta','5':'tab:pink'}

dat={}
dir="/home/shared/cases/nozzle/eigenvalues/"
file_names = [f for f in listdir(dir+"values/") if f[-3:]=="txt"]
for file_name in file_names:
    match = search(r'm=(-?\d*)', file_name)
    m=match.group(1)
    match = search(r'Re=(\d*)',file_name)
    Re=match.group(1)
    match = search(r'S=(\d*\,?\d*)',file_name)
    S=match.group(1)
    if not Re in dat.keys(): 		dat[Re]      ={}
    if not S  in dat[Re].keys():	dat[Re][S]   ={}
    if not m  in dat[Re][S].keys(): dat[Re][S][m]=set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local_eigs=np.loadtxt(dir+"values/"+file_name,dtype=complex)
    if local_eigs.size==1: dat[Re][S][m].add(complex(local_eigs))
    else:                  dat[Re][S][m].update(local_eigs)
dirCreator(dir+"plots/")
#plt.rcParams.update({'font.size': 26})
for Re in dat.keys():
    for S in dat[Re].keys():
        for m in dat[Re][S].keys():
            eigs=np.array(list(dat[Re][S][m]),dtype=complex)
            # Plot them all!
            fig = plt.figure(dpi=200)
            msk=eigs.real<0
            plt.scatter(eigs.imag[msk], eigs.real[msk], edgecolors=color_code[m],facecolors='none')  # Stable eigenvalues
            if eigs[~msk].size>0:
                plt.scatter(eigs.imag[~msk],eigs.real[~msk],edgecolors=color_code[m],facecolors=color_code[m]) # Unstable eigenvalues
            #plt.plot([-np.min(eigs.imag),np.max(eigs.imag)],[0,0],'k--')
            #plt.axis([-2,2,-2,2])
            plt.xlabel(r'$\omega$')
            plt.ylabel(r'$\sigma$')
            plt.savefig(dir+f"plots/eigenvalues_Re={Re}_S={S}_m={m}".replace('.',',')+".png")
            plt.close()