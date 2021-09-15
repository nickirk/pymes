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

n_elec, n_orb, e_core, epsilon, h, V_pqrs = read_fcidump("FCIDUMP.test")

print("Checking known integral values")
V_000 = V_pqrs[0,0,0,0]
assert(np.abs(V_000-0.3477076890057987E+00) < 1e-12)
# 10   8   7   7
V_9676 = V_pqrs[9,6,7,6]
assert(np.abs(V_9676 - -0.7435116502951884E-03) < 1e-12)

# epsilon is all 0

# h
h_82 = h[8,2]
assert(np.abs(h_82 - 0.3365437441949438E-01) < 1e-12)

assert(np.abs(e_core - 0.1071649029982363E+02) < 1e-12)

print("Passed")


t_V_pqrs = ctf.astensor(V_pqrs)

sym = ctf.einsum("pqrs -> qprs", t_V_pqrs) - t_V_pqrs
sym = ctf.einsum("pqrs ->", sym)
print_logging_info("Exchange of 1 and 2 indices = ", sym)

sym = ctf.einsum("pqrs -> rqps", t_V_pqrs) - t_V_pqrs
sym = ctf.einsum("pqrs ->", sym)
print_logging_info("Exchange of 1 and 3 indices= ", sym)

sym = ctf.einsum("pqrs -> sqrp", t_V_pqrs) - t_V_pqrs
sym = ctf.einsum("pqrs ->", sym)
print_logging_info("Exchange of 1 and 4 indices= ", sym)

sym = ctf.einsum("pqrs -> prqs", t_V_pqrs) - t_V_pqrs
sym = ctf.einsum("pqrs ->", sym)
print_logging_info("Exchange of 2 and 3 indices= ", sym)

sym = ctf.einsum("pqrs -> psrp", t_V_pqrs) - t_V_pqrs
sym = ctf.einsum("pqrs ->", sym)
print_logging_info("Exchange of 2 and 4 indices= ", sym)

sym = ctf.einsum("pqrs -> pqsr", t_V_pqrs) - t_V_pqrs
sym = ctf.einsum("pqrs ->", sym)
print_logging_info("Exchange of 3 and 4 indices= ", sym)
