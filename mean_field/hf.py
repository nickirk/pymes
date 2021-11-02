import time
import numpy as np
import ctf
from ctf.core import *

def calc_hf_e(no, e_core, t_h_pq, t_V_pqrs):

    t_e_hf = 2.* ctf.einsum('ii->', t_h_pq[:no,:no])
    dirHFE = 2. * ctf.einsum('jiji->',t_V_pqrs[:no,:no,:no,:no])
    excHFE = -1. * ctf.einsum('ijji->',t_V_pqrs[:no,:no,:no,:no])
    
    t_e_hf = t_e_hf + (dirHFE + excHFE) + e_core
    return t_e_hf

def construct_hf_matrix(no, t_h_pq, t_V_pqrs):
    t_fock_pq = t_h_pq.copy()
    t_fock_pq += ctf.einsum("piqi -> pq", 2.*t_V_pqrs[:,:no,:,:no])\
                 -ctf.einsum("piiq -> pq", t_V_pqrs[:,:no,:no,:])

    return t_fock_pq


def calcOccupiedOrbE(kinetic_G, tV_ijkl, no):
    dtype = tV_ijkl.dtype
    e = ctf.astensor(kinetic_G[0:no], dtype = dtype)
    #tConjGamma_jiG = ctf.einsum("ijG->jiG", ctf.conj(tGamma_ijG))
    #coul = ctf.einsum('ikG,jlG->ijkl', tConjGamma_jiG, tGamma_ijG)
    #exCoul = ctf.einsum('ikG,ljG->ilkj', tConjGamma_jiG, tGamma_ijG)
    dirE = 2.* ctf.einsum('ijij->i', tV_ijkl)
    exE = - 1.* ctf.einsum('ijji->i', tV_ijkl)
    e = e + dirE 
    e = e + exE
    return e

def calcVirtualOrbE(kinetic_G, t_V_aibj, t_V_aijb, no, nv):
    algoName = "calcVirtualOrbE"
    e = ctf.astensor(kinetic_G[no:], dtype = t_V_aijb.dtype)
    #tConjGamma_aiG = ctf.einsum("iaG -> aiG", ctf.conj(tGamma_iaG))
    #dirCoul_aibj =  ctf.einsum('aiG,bjG->aibj',tConjGamma_aiG, tGamma_aiG)
    #exCoul_aijb = ctf.einsum('ajG,ibG->aijb',tConjGamma_aiG, tGamma_iaG)
    dirE = ctf.tensor([nv], dtype=t_V_aijb.dtype, sp=0)
    dirE.i("a") << 2. * t_V_aibj.i("aiai")
    exE = ctf.tensor([nv], dtype=t_V_aibj.dtype, sp=0)
    exE.i("a") << -1. * t_V_aijb.i("aiia")

    e = e + dirE
    e = e + exE
    return e
