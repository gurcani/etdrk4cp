#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 20:47:15 2025

@author: ogurcan
"""

import cupy as xp
#import numpy as xp # (also change in the etdrk4cp.py)
import h5py as h5
from time import time
from h5tools import save_data
from gsol import gsol,callbacks
import os

Np=2**18
nu=1e-4
t0,t1=0.0,100.0
dtstep=1e-5
flname='out.h5'
N=int(Np/3)*2
Nh=int(N/2)+1
k=xp.arange(0,int(N/2)+1)
u0=1e-6*xp.exp(-k**2/100**2+1j*2*xp.pi*(xp.random.rand(*k.shape)))
force_k=1e2*(k>2)*(k<10)*xp.exp(-k**2/100**2+1j*2*xp.pi*(xp.random.rand(*k.shape)))/100
ct=0

get = lambda x : x.get() if (xp.__name__ == 'cupy') else x

def fftconv(gk,hk):
    gkp=xp.zeros(int(Np/2)+1,dtype=gk.dtype)
    hkp=xp.zeros(int(Np/2)+1,dtype=gk.dtype)
    gkp[:Nh]=gk
    hkp[:Nh]=hk
    fkp=xp.fft.rfft(xp.fft.irfft(gkp,norm='forward')*xp.fft.irfft(hkp,norm='forward'),norm='forward')
    fk=fkp[:Nh]
    return fk

def rhsNl(t,uk):
    dukdt=xp.zeros_like(uk)
    dukdt[:]=(-fftconv(uk,1j*k*uk)+force_k)
    return dukdt

def force_update(t,uk):
    force_k[:]=force_k*xp.exp(1j*2*xp.pi*(xp.random.rand(*force_k.shape)))

def save_callback(t,uk):
    save_data(fl,'fields',ext_flag=True,t=t,uk=get(uk))

def showcb(t,uk):
    print('t=',t,', ',time()-ct,' secs elapsed')

cbs = callbacks([0.01,0.1,0.01],[force_update,save_callback,showcb])
r=gsol(rhsNl,t0,u0,t1,-nu*k**2,dtstep,callbacks=cbs,tol=1e-7)

if os.path.exists(flname):
    os.remove(flname)
fl=h5.File(flname,'w',libver='latest')
fl.swmr_mode = True
save_data(fl, 'data', nu=nu,k=get(k))
save_data(fl,'fields',ext_flag=True,t=t0,uk=get(u0))
ct=time()
r.run()
