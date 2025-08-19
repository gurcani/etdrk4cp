#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 22:14:14 2025

@author: ogurcan
"""
from etdrk4cp import etdrk4cp as etd

class callbacks:
    def __init__(self,dts,fncbs):
        self.fncbs=fncbs
        self.dts=dts
        self.ts=[0.0 for l in range(len(dts)) ]
        self.tnexts=dts.copy()
    def append(self,dt,fncb,t=0):
        self.fncbs.append(fncb)
        self.dts.append(dt)
        self.ts.append(t)
        self.tnexts.append(dt)
    def act(self,t,u):
        for l in range(len(self.dts)):
            if(t>=self.tnexts[l]):
                self.fncbs[l](t,u)
                self.tnexts[l]+=self.dts[l]

class gsol:
    def __init__(self,fexp,t0,y0,t1,L,dtstep,callbacks=None,tol=1e-7,**kwargs):
        self.t0=float(t0)
        self.t=float(t0)
        self.t1=float(t1)
        self.y=y0
        self.h=dtstep
        self.tol=tol
        if (L.size==y0.size): # means L is diagonal
            self.r_=etd(fexp,L,y0,self.h,**kwargs)
        #else: # diagonalize and use the eigenmodes as Ldia
        self.cbs=callbacks

    def run(self,):
        r=self.r_
        while(self.t<self.t1):
            uk,err=r.step(self.t,self.y)
            self.y[:]=uk
            self.t+=r.h
            self.cbs.act(self.t,self.y)
            r.recompute_stepsize(err,self.tol)
