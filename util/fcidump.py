#!/usr/bin/python3
import os.path
import ctf
import numpy as np

from pymes.logging import print_logging_info


def write(integrals, kinetic, no, ms2=1, orbsym=1, isym=1, dtype='r'):
    """
    Input a integral file in ctf format
    np,np,np,np
    """
    world = ctf.comm()

    nP = integrals.shape[0]
    inds, vals = integrals.read_all_nnz()
    if world.rank() == 0:
        f = open("FCIDUMP", "w")
        # write header
        f.write("&FCI\n")
        f.write(" NORB=%i,\n" % nP)
        f.write(" NELEC=%i,\n" % (no * 2))
        f.write(" MS2=%i,\n" % ms2)
        # prepare orbsym
        OrbSym = [orbsym] * nP
        f.write(" ORBSYM=" + str(OrbSym).strip('[]') + ",\n")
        f.write(" ISYM=%i,\n" % isym)
        f.write("/\n")

        for l in range(len(inds)):
            p = int(inds[l] / nP ** 3)
            q = int((inds[l] - p * nP ** 3) / nP ** 2)
            r = int((inds[l] - p * nP ** 3 - q * nP ** 2) / nP)
            s = int(inds[l] - p * nP ** 3 - q * nP ** 2 - r * nP)

            f.write("  " + str(vals[l]) + "  " + str(p + 1) \
                    + "  " + str(r + 1) + "  " + str(q + 1) + "  " + str(s + 1) + "\n")

        for i in range(nP):
            f.write("  " + str(kinetic[i]) + "  " + str(i + 1) + "  " \
                    + str(i + 1) + "  0  0\n")

        # for i in range(nP-no):
        #    f.write("  " + str(particleEnergies[i]) + "  " + str(i+no+1)\
        #            + "  0  0  0\n")

        f.write("  0.0  0  0  0  0")

        f.close()
    return


def read(fcidump_file="FCIDUMP", is_tc=False):
    """
    Read Coulomb integrals from a FCIDUMP file. Works only on a single rank
    and for small FCIDUMP files ~300 orbitals for 120 GB RAM assuming dense
    tensor.

    Parameter:
    ---------
    fcidump_file: string
                  filename/path to the FCIDUMP file to be read.
    is_tc: bool
           tells if the FCIDUMP file is symmetric or not (transcorrelated or not)
    Return:
    ------
    n_elec: int
            number of electrons
    n_orb: int
           number of orbitals
    epsilon_p: numpy tensor, [n_orb]
                 the orbital energies.
    h_pq: numpy tensor, [n_orb, n_orb]
            the single operator values.
    V_pqrs: numpy tensor, [nb, nb, nb, nb]
              the Coulomb integrals.
    """
    world = ctf.comm()

    header_dict = {"norb": 0, "nelec": 0}

    try:
        os.path.exists(fcidump_file)
    except FileNotFoundError:
        sys.exit(1)

    print_logging_info("Reading " + fcidump_file + "...", level=1)

    e_core = 0.

    # if world.rank() == 0:
    with open(fcidump_file, 'r') as reader:

        print_logging_info("Parsing header...", level=2)
        line = reader.readline().strip()

        while not (('/' in line) or ("END".lower() in line.lower())):
            line += reader.readline().strip()

        header = line.split(",")

        for key in header_dict.keys():
            for attr in header:
                if key in attr.lower():
                    for word in attr.split("="):
                        word = word.strip()
                        if word.isdigit():
                            header_dict[key] = int(word)
                            continue
                    continue
        n_elec = header_dict["nelec"]
        n_orb = header_dict["norb"]
        epsilon_p = np.zeros(n_orb)
        h_pq = np.zeros([n_orb, n_orb])
        V_pqrs = np.zeros([n_orb, n_orb, n_orb, n_orb])

        print_logging_info("Reading integrals...", level=2)
        while True:
            line = reader.readline()
            # if not line.strip():
            #    continue
            if not line:
                break
            integral, p, r, q, s = line.split()
            integral = float(integral)
            # pqrs
            p = int(p)
            r = int(r)
            q = int(q)
            s = int(s)

            if np.abs(integral) < 1e-19:
                continue

            if p != 0 and q != 0 and r != 0 and s != 0:
                if not is_tc:
                    V_pqrs[p - 1, q - 1, r - 1, s - 1] = integral
                    V_pqrs[r - 1, q - 1, p - 1, s - 1] = integral
                    V_pqrs[r - 1, s - 1, p - 1, q - 1] = integral
                    V_pqrs[p - 1, s - 1, r - 1, q - 1] = integral
                else:
                    V_pqrs[p - 1, q - 1, r - 1, s - 1] = integral

            if p == q == r == s == 0:
                e_core = integral

            if p != 0 and q == r == s == 0:
                epsilon_p[p - 1] = integral

            if p != 0 and r != 0 and q == s == 0:
                if not is_tc:
                    h_pq[p - 1, r - 1] = integral
                    h_pq[r - 1, p - 1] = integral
                else:
                    h_pq[p - 1, r - 1] = integral

    return n_elec, n_orb, e_core, epsilon_p, h_pq, V_pqrs
