#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:19:37 2024

@author: ogurcan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 12:02:31 2025

@author: ogurcan
"""

import numpy as np
import cupy as xp
from cupyx.scipy.fft import rfft2,irfft2
from gsol import gsol,callbacks
from h5tools import save_data
import h5py as h5
from time import time
import os

get = lambda x : x.get() if (xp.__name__ == 'cupy') else x
dot = lambda x,y : np.einsum('ijk,ki->ji',x,y)

flname='out.h5'
Npx,Npy=1024,1024
t0,t1=0,300.0
wecontinue=False
Lx,Ly=12*np.pi,12*np.pi
kap=1.0
C=1.0
nu=5e-4
D=5e-4
Nx,Ny=2*int(np.floor(Npx/3)),2*int(np.floor(Npy/3))

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

dkx,dky=2*np.pi/Lx,2*np.pi/Ly
sl=slicelist(Nx,Ny)
lkx,lky=init_kspace_grid(sl)
kx,ky=lkx*dkx,lky*dky
ksqr=kx**2+ky**2
sigk=(ky>0)
w=10.0
zk0=xp.zeros((2,kx.size),dtype=complex)
zk0[0,:]=1e-4*xp.exp(-lkx**2/2/w**2-lky**2/w**2)*xp.exp(1j*2*xp.pi*xp.random.rand(lkx.size).reshape(lkx.shape));
zk0[1,:]=1e-4*xp.exp(-lkx**2/w**2-lky**2/w**2)*xp.exp(1j*2*xp.pi*xp.random.rand(lkx.size).reshape(lkx.shape));
xl,yl=np.arange(0,Lx,Lx/Npx),np.arange(0,Ly,Ly/Npy)
x,y=np.meshgrid(np.array(xl),np.array(yl),indexing='ij')

Lk=np.zeros(kx.shape+(2,2),dtype=complex)
Lk[:,0,0]=get(-C*sigk/ksqr-nu*ksqr*sigk)
Lk[:,0,1]=get(C*sigk/ksqr)
Lk[:,1,0]=get(-1j*kap*ky+C*sigk)
Lk[:,1,1]=get(-C*sigk-D*ksqr*sigk)
gams,Tk=np.linalg.eig(Lk)
gams=xp.array(gams)
Tinvk=np.linalg.inv(Tk)
Tk=xp.array(Tk)
Tinvk=xp.array(Tinvk)

def irft(uk):
    u=mlsarray(Npx,Npy)
    u[sl]=uk
    u[-1:-int(Nx/2):-1,0]=u[1:int(Nx/2),0].conj()
    u.view(dtype=float)[:,:-2]=irfft2(u,norm='forward',overwrite_x=True)
    return u.view(dtype=float)[:,:-2]

def rft(u):
    uk=rfft2(u,norm='forward',overwrite_x=True).view(type=mlsarray)
    return np.hstack(uk[sl])

def save_callback(fl,t,zk,flag):
    xik=zk.reshape((2,kx.size))
    save_data(fl,'last',ext_flag=False,zk=get(zk),t=t)
    phink=dot(Tk,xik)
    phik,nk=phink[0,:],phink[1,:]
    if flag=='fields':
        print('saving fields')
        om=irft(-phik*(kx**2+ky**2))
        n=irft(nk)
        save_data(fl,'fields',ext_flag=True,om=get(om),n=get(n),t=t)
    if flag=='energies':
        print('saving energies')
        Etot=xp.sum(xp.abs(phik)**2*(kx**2+ky**2))
        Ez=xp.sum(xp.abs(phik)**2*(kx**2+ky**2)*(ky==0))
        Ftot=xp.sum(xp.abs(nk)**2)
        Fz=xp.sum(xp.abs(nk**2)*(ky==0))
        save_data(fl,'energies',ext_flag=True,Etot=get(Etot),Ez=get(Ez),Ftot=get(Ftot),Fz=get(Fz),t=t)

def rhs_phi(t,zk):
    dzkdt=xp.zeros_like(zk)
    phik,nk=zk[0,:],zk[1,:]
    dphikdt,dnkdt=dzkdt[0,:],dzkdt[1,:]
    dxphi=irft(1j*kx*phik)
    dyphi=irft(1j*ky*phik)
    om=irft(-ksqr*phik)
    n=irft(nk)
    dphikdt[:]=(-1j*kx*rft(dyphi*om)+1j*ky*rft(dxphi*om))/ksqr
    dnkdt[:]=1j*kx*rft(dyphi*n)-1j*ky*rft(dxphi*n)
    return dzkdt

def rhs2(t,zk):
    xik=zk.reshape((2,kx.size))
    phink=dot(Tk,xik)
    dphinkdt=rhs_phi(t,phink)
    dzkdt=dot(Tinvk,dphinkdt)
    return dzkdt.ravel()

if(wecontinue):
    fl=h5.File(flname,'r+',libver='latest')
    fl.swmr_mode = True
    omk,nk=rft(np.array(fl['fields/om'][-1,])),rft(np.array(fl['fields/n'][-1,]))
    phik=-omk/(kx**2+ky**2)
    t0=fl['fields/t'][-1]
    zk=xp.hstack((phik,nk))
else:
    if os.path.exists(flname):
        os.remove(flname)
    fl=h5.File(flname,'w',libver='latest')
    fl.swmr_mode = True
    save_data(fl,'data',ext_flag=False,x=x,y=y,kap=kap,C=C,nu=nu,D=D)

ct=time()
fcbs = [(lambda t,y : print('t=',t,', ',time()-ct,' secs elapsed')),
        (lambda t,y : save_callback(fl,t,y,flag='fields')),
         (lambda t,y : save_callback(fl,t,y,flag='energies'))]
dtstep=1.0
dtcbs=[1.0,1.0,1.0]
cbs=callbacks(dtcbs,fcbs)
ksqr=kx**2+ky**2
sigk=(ky>0)
L=gams.T.ravel()
r=gsol(rhs2,t0,zk0.ravel(),t1,L,dtstep,callbacks=cbs,tol=1e-7,M=16,maxstep=1.0)
r.run()
fl.close()
