#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-

import time
import sys
import os.path
sys.path.append("/home/liao/Work/Research/")

import ctf
import numpy as np

import pymes
from pymes.util.fcidump import read_fcidump
from pymes.mean_field import hf
from pymes.logging import print_logging_info

print_logging_info("Testing tc-FCIDUMP reader...")
n_elec, n_orb, e_core, epsilon, h, V_pqrs = read_fcidump("./test_contraction/FCIDUMP", is_tc=True)


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