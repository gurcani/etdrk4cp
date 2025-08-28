#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:19:37 2024

@author: ogurcan
"""

import cupy as xp
import numpy as np

import sys
from gsol import gsol,callbacks
from h5tools import save_data
import h5py as h5
from scipy.special import j0,j1,jn_zeros
from time import time
import os
from cupyx.scipy.fft import rfft2,irfft2
get = lambda x : x.get() if (xp.__name__ == 'cupy') else x

flname="outns.h5"
Npx,Npy=1024,1024
t0,t1=0.0,500.0
Nx,Ny=2*int(np.floor(Npx/3)),2*int(np.floor(Npy/3))
Lx,Ly=2*np.pi,2*np.pi

class slicelist:
    def __init__(self,Nx,Ny):
        shp=(Nx,Ny)
        insl=[np.s_[0:1,1:int(Ny/2)],np.s_[1:int(Nx/2),:int(Ny/2)],np.s_[-int(Nx/2)+1:,1:int(Ny/2)]]
        shps=[[len(range(*(l[j].indices(shp[j])))) for j in range(len(l))] for l in insl]
        Ns=[np.prod(l) for l in shps]
        outsl=[np.s_[sum(Ns[:l]):sum(Ns[:l])+Ns[l]] for l in range(len(Ns))]
        self.insl,self.shape,self.shps,self.Ns,self.outsl=insl,shp,shps,Ns,outsl

class mlsarray(xp.ndarray):
    def __new__(cls,Nx,Ny):
        v=xp.zeros((Nx,int(Ny/2)+1),dtype=complex).view(cls)
        return v
    def __getitem__(self,key):
        if(isinstance(key,slicelist)):
            return [xp.ndarray.__getitem__(self,l).ravel() for l in key.insl]
        else:
            return xp.ndarray.__getitem__(self,key)
    def __setitem__(self,key,value):
        if(isinstance(key,slicelist)):
            for l,j,shp in zip(key.insl,key.outsl,key.shps):
                self[l]=value.ravel()[j].reshape(shp)
        else:
            xp.ndarray.__setitem__(self,key,value)

def init_kspace_grid(sl):
    Nx,Ny=sl.shape
    kxl=np.r_[0:int(Nx/2),-int(Nx/2):0]
    kyl=np.r_[0:int(Ny/2+1)]
    kx,ky=np.meshgrid(kxl,kyl,indexing='ij')
    kx=xp.hstack([kx[l].ravel() for l in sl.insl])
    ky=xp.hstack([ky[l].ravel() for l in sl.insl])
    return kx,ky

def init_forcing(kx,ky,A=1e2,k0=50.0,dk=2.0):
    k=xp.sqrt(kx**2+ky**2)
    return A*xp.exp(-(k-k0)**2/2/dk**2+1j*2*xp.pi*xp.random.rand(k.size))/k.size

def update_forcing(fk):
    print("updating forcing...")
    fk[:]=fk*xp.exp(1j*2*xp.pi*xp.random.rand(fk.size))
    
def irft(uk):
    u=mlsarray(Npx,Npy)
    u[sl]=uk
    u[-1:-int(Nx/2):-1,0]=u[1:int(Nx/2),0].conj()
    u.view(dtype=float)[:,:-2]=irfft2(u,norm='forward',overwrite_x=True)
    return u.view(dtype=float)[:,:-2]

def rft(u):
    uk=rfft2(u,norm='forward',overwrite_x=True).view(type=mlsarray)
    return xp.hstack(uk[sl])

def save_callback(fl,t,y):
    phik=y.view(dtype=complex)
    om=irft(-phik*(kx**2+ky**2))
    save_data(fl,'fields',ext_flag=True,om=get(om),t=t)

def rhs(t,y):
    phik=y.view(dtype=complex)
    dphikdt=xp.zeros_like(phik)
    dxphi=irft(1j*kx*phik)
    dyphi=irft(1j*ky*phik)
    om=irft(-ksqr*phik)
    dphikdt[:]=-1j*kx*rft(dyphi*om)/ksqr+1j*ky*rft(dxphi*om)/ksqr
    dphikdt[:]+=+fk[:]
    return dphikdt

dkx,dky=2*np.pi/Lx,2*np.pi/Ly
sl=slicelist(Nx,Ny)
lkx,lky=init_kspace_grid(sl)
kx,ky=lkx*dkx,lky*dky
xl,yl=np.arange(-Lx/2,Lx/2,Lx/Npx),np.arange(-Ly/2,Ly/2,Ly/Npy)
x,y=np.meshgrid(xl,yl,indexing='ij')
ksqr=kx**2+ky**2
fk=init_forcing(kx,ky)

phik=1e-8*fk.copy()
nu=1e-4
nuL=1e1
om=irft(-ksqr*phik)
if os.path.exists(flname):
    os.remove(flname)
fl=h5.File(flname,'w',libver='latest')
fl.swmr_mode = True
save_data(fl,'data',ext_flag=False,x=x,y=y)
save_data(fl,'fields',ext_flag=True,om=get(om),t=t0)
ct=time()
fcbs = [(lambda t,y : print('t=',t,', ',time()-ct,' secs elapsed')),
        (lambda t,y : update_forcing(fk)),
        (lambda t,y : save_callback(fl,t,y))]
dtstep=0.1
dtcbs = [0.1,0.1,1.0]
cbs = callbacks(dtcbs,fcbs)
r=gsol(rhs,t0,phik,t1,-nu*ksqr-nuL/ksqr**3,dtstep,callbacks=cbs,tol=1e-9,maxstep=0.1)

#r=gensolver('cupy_ivp.DOP853',rhs,t0,phik.view(dtype=float),t1,fsave=fsave,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,rtol=1e-8,atol=1e-12)
r.run()
fl.close()
