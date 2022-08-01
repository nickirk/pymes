#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-


import ctf
import numpy as np
from pymes.log import print_logging_info
from pymes.util.fcidump import read, write


def test_tc_fcidump_reader():
    print_logging_info("Testing tc-FCIDUMP reader...")
    n_elec, n_orb, e_core, epsilon, h, V_pqrs = read("../test_tc_ccsd/FCIDUMP.LiH.tc", is_tc=True)


    t_V_pqrs = ctf.astensor(V_pqrs)
    t_V_ex_pqrs = t_V_pqrs.copy()

    print_logging_info("Exchange 2 electron indices")
    sym = ctf.einsum("pqrs -> qpsr", t_V_ex_pqrs) - t_V_pqrs
    sym = ctf.einsum("pqrs ->", ctf.abs(sym))
    print_logging_info("Exchange of (pr) and (qs)=", sym)
    assert sym < 1.e-12, "Tensor does not have this symmetry!"
#
    sym = ctf.einsum("pqrs -> rqps", t_V_ex_pqrs) - t_V_pqrs
    sym = ctf.einsum("pqrs ->", ctf.abs(sym))
    assert sym > 1.e-12, "Tensor should not have this symmetry!"

    print_logging_info("Exchange of p and r indices= ", sym)
#
    sym = ctf.einsum("pqrs -> sqrp", t_V_ex_pqrs) - t_V_pqrs
    sym = ctf.einsum("pqrs ->", ctf.abs(sym))
    print_logging_info("Exchange of p and s indices= ", sym)
    assert sym > 1.e-12, "Tensor should not have this symmetry!"
#
    sym = ctf.einsum("pqrs -> prqs", t_V_ex_pqrs) - t_V_pqrs
    sym = ctf.einsum("pqrs ->", ctf.abs(sym))
    print_logging_info("Exchange of q and r indices= ", sym)
    assert sym > 1.e-12, "Tensor should not have this symmetry!"

    sym = ctf.einsum("pqrs -> psrp", t_V_ex_pqrs) - t_V_pqrs
    sym = ctf.einsum("pqrs ->", ctf.abs(sym))
    print_logging_info("Exchange of q and s indices= ", sym)
    assert sym > 1.e-12, "Tensor should not have this symmetry!"

    sym = ctf.einsum("pqrs -> pqsr", t_V_ex_pqrs) - t_V_pqrs
    sym = ctf.einsum("pqrs ->", ctf.abs(sym))
    print_logging_info("Exchange of s and r indices= ", sym)
    assert sym > 1.e-12, "Tensor should not have this symmetry!"
    print_logging_info("All tests passed!")

def test_fcidump_write():
    n_elec, n_orb, e_core, epsilon, h, V_pqrs = read("../test_tc_ccsd/FCIDUMP.LiH.tc", is_tc=True)
    no = n_elec // 2
    write(ctf.astensor(V_pqrs), ctf.astensor(h), no, e_core, file="fcidump.w")
    n_elec_r, n_orb_r, e_core_r, epsilon_r, h_r, V_pqrs_r = read("./fcidump.w", is_tc=True)
    assert n_elec_r == n_elec
    assert n_orb_r == n_orb
    assert e_core_r == e_core
    assert np.array_equal(epsilon_r, epsilon)
    assert np.array_equal(h_r, h)
    assert np.array_equal(V_pqrs_r, V_pqrs)