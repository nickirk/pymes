import time
import numpy as np
import ctf
from ctf.core import *



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

def calcVirtualOrbE(kinetic_G, tV_aibj, tV_aijb, no, nv):
    algoName = "calcVirtualOrbE"
    e = ctf.astensor(kinetic_G[no:], dtype = tV_aijb.dtype)
    #tConjGamma_aiG = ctf.einsum("iaG -> aiG", ctf.conj(tGamma_iaG))
    #dirCoul_aibj =  ctf.einsum('aiG,bjG->aibj',tConjGamma_aiG, tGamma_aiG)
    #exCoul_aijb = ctf.einsum('ajG,ibG->aijb',tConjGamma_aiG, tGamma_iaG)
    dirE = ctf.tensor([nv], dtype=tV_aijb.dtype, sp=0)
    dirE.i("a") << 2. * tV_aibj.i("aiai")
    exE = ctf.tensor([nv], dtype=tV_aibj.dtype, sp=0)
    exE.i("a") << -1. * tV_aijb.i("aiia")

    e = e + dirE
    e = e + exE
    return e
