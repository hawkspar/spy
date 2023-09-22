# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from sys import path
from re import search
from os import listdir
from matplotlib import pyplot as plt
from warnings import simplefilter, catch_warnings

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from handlers import *
from helpers import dirCreator

# Shorthands
x_lims=[-1e-1,1e-1]
y_lims=[-1e-2,1e-2]

dat={}
dir="/home/shared/cases/nozzle/eigenvalues/"
file_names = [f for f in listdir(dir+"values/") if f[-3:]=="txt"]
for file_name in listdir(dir+"values/"):
    if file_name[-3:]=="txt":
        match = search(r'm=(-?\d*)', file_name)
        m=int(match.group(1))
        match = search(r'Re=(\d*)',file_name)
        Re=int(match.group(1))
        match = search(r'S=(\d*\,?\d*)',file_name)
        S=float(match.group(1).replace(',','.'))
        if not Re in dat.keys(): 		dat[Re]      ={}
        if not S  in dat[Re].keys():	dat[Re][S]   ={}
        if not m  in dat[Re][S].keys(): dat[Re][S][m]=set()
        with catch_warnings():
            simplefilter("ignore")
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
            plt.xlim(x_lims)
            plt.ylim(y_lims)
            plt.gca().set_aspect('equal')
            plt.xlabel(r'$\omega$')
            plt.ylabel(r'$\sigma$')
            plt.savefig(dir+f"plots/eigenvalues_S={S}_m={m}".replace('.',',')+".png") # Usual Re is based on D
            plt.close()