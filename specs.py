#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:47:50 2023

@author: ogurcan
"""
import cupy as xp
import h5py as h5
import numpy as np
import matplotlib.pylab as plt
from ns2d_forced import mlsarray,slicelist,init_kspace_grid,rfft2
get = lambda x : x.get() if (xp.__name__ == 'cupy') else x

flname="outns.h5"
fl=h5.File(flname,'r',swmr=True)
om=fl['fields/om']
t=fl['fields/t'][()]
Npx,Npy=om.shape[1:]
Nx,Ny=2*int(np.floor(Npx/3)),2*int(np.floor(Npy/3))
Lx,Ly=2*np.pi,2*np.pi
dkx,dky=2*np.pi/Lx,2*np.pi/Ly
sl=slicelist(Nx,Ny)
rft = lambda u : xp.hstack(rfft2(u,norm='forward',overwrite_x=True).view(type=mlsarray)[sl])
lkx,lky=init_kspace_grid(sl)
kx,ky=lkx*dkx,lky*dky
ksqr=kx**2+ky**2

ts=np.arange(100,200,2)
tinds=[np.argmax(t>=ts[i]) for i in range(ts.size)]
tinds.append(t.size-1)
N=int(Ny/2-1)
En=xp.zeros((len(tinds),N))
kn=xp.arange(N)
dk=1
k=xp.sqrt(ksqr)
filt=xp.zeros((N,)+kx.shape)
print("preparing filter...")
for l in range(N):
    filt[l,]=1*((k<=kn[l]+dk/2) & (k>kn[l]-dk/2) & ((kx>=0) | (ky>0)))
print("reading data")
print("computing einsum")
for l in range(len(tinds)):
    phik=-rft(xp.array(om[tinds[l],]))/ksqr
    En[l,]=np.einsum('k,jk->j',ksqr*np.abs(phik)**2,filt)

kn=get(kn)
En=get(En)
plt.figure(figsize=(9,8))
plt.loglog(kn,np.mean(En,0),'x-',kn,En[0,],'lightgray',kn,En[-1,],'gray',kn,kn**(-5/3),'C1')
plt.rcParams.update({'font.size': 14})
plt.ylabel('$E(k)$')
plt.xlabel('$k$')
plt.text(kn[50],kn[50]**(-5/3)*2,'$k^{-5/3}$',color='C1')
plt.legend(['mean [150-200]','$t=150$','$t=200$'])
plt.tight_layout()
plt.savefig('out.png')
fl.close()
