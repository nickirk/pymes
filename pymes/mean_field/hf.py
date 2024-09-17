import time
import numpy as np


def calc_hf_e(no, e_core, t_h_pq, t_V_pqrs):
    t_e_hf = 2. * np.einsum('ii->', t_h_pq[:no, :no])
    dirHFE = 2. * np.einsum('jiji->', t_V_pqrs[:no, :no, :no, :no])
    excHFE = -1. * np.einsum('ijji->', t_V_pqrs[:no, :no, :no, :no])

    t_e_hf = t_e_hf + (dirHFE + excHFE) + e_core
    return t_e_hf


def construct_hf_matrix(no, t_h_pq, t_V_pqrs):
    t_fock_pq = t_h_pq.copy()
    t_fock_pq += 2. * np.einsum("piqi -> pq", t_V_pqrs[:, :no, :, :no])
    t_fock_pq += -1. * np.einsum("piiq -> pq", t_V_pqrs[:, :no, :no, :])
    return t_fock_pq


def calcOccupiedOrbE(kinetic_G, tV_ijkl, no):
    e = kinetic_G[0:no]
    # tConjGamma_jiG = np.einsum("ijG->jiG", np.conj(tGamma_ijG))
    # coul = np.einsum('ikG,jlG->ijkl', tConjGamma_jiG, tGamma_ijG)
    # exCoul = np.einsum('ikG,ljG->ilkj', tConjGamma_jiG, tGamma_ijG)
    dirE = 2. * np.einsum('ijij->i', tV_ijkl)
    exE = - 1. * np.einsum('ijji->i', tV_ijkl)
    e = e + dirE
    e = e + exE
    return e


def calcVirtualOrbE(kinetic_G, t_V_aibj, t_V_aijb, no, nv):
    e = kinetic_G[no:]
    # tConjGamma_aiG = np.einsum("iaG -> aiG", np.conj(tGamma_iaG))
    # dirCoul_aibj =  np.einsum('aiG,bjG->aibj',tConjGamma_aiG, tGamma_aiG)
    # exCoul_aijb = np.einsum('ajG,ibG->aijb',tConjGamma_aiG, tGamma_iaG)
    dirE = 2. * np.einsum("aiai -> a", t_V_aibj)
    exE =  -1. * np.einsum("aiia -> a", t_V_aijb)

    e = e + dirE
    e = e + exE
    return e
