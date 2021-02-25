import time
import warnings

import numpy as np
from pymes.basis_set import planewave
from pymes.logging import print_logging_info
import ctf
from scipy import special



class UEG:

    def __init__(self, nel, nalpha, nbeta, rs):
        #: number of electrons
        if (nel) % 2 != 0:
            warnings.warn("The number of electrons is not even, currently only\
                          closed shell systems are supported!")
        self.nel = int(nel)
        #: number of alpha (spin-up) electrons
        self.nalpha = int(nalpha)
        #: number of beta (spin-down) electrons
        self.nbeta = int(nbeta)
        if self.nalpha != self.nbeta:
            warnings.warn("The number of electrons is not even, currently only\
                          closed shell systems are supported!")
        #: electronic density
        self.rs = rs
        #: length of the cubic simulation cell containing nel electrons
        #: at the density of rs
        self.L = self.rs*((4*np.pi*self.nel)/3)**(1.0/3.0)
        #: volume of the cubic simulation cell containing nel electrons at the density
        #: of rs
        self.Omega = self.L**3

        self.basis_fns = None

        self.imax = 0

        self.cutoff = 0.

        self.basis_indices_map = None

        self.kPrime = None

        self.correlator = None

        self.kCutoff = None

        self.gamma = None


    def is_k_in_basis(self,k):
        if k.dot(k) <= self.cutoff:
            return True
        return False

    def init_basis_indices_map(self):
        numKInEachDir = self.imax*2+1
        self.basis_indices_map = -1 * np.ones(numKInEachDir**3).astype(int)
        for i in range(int(len(self.basis_fns)/2)):
            s = numKInEachDir**2 * (self.basis_fns[i*2].k[0] + self.imax) + \
                    numKInEachDir * (self.basis_fns[i*2].k[1] + self.imax) +\
                    self.basis_fns[i*2].k[2] + self.imax
            self.basis_indices_map[s] = i


    #--- Basis set ---
    def init_single_basis(self, cutoff):
        '''Create single-particle basis.

    :type self: :class:`UEG`
    :param sys: UEG system to be studied.

    :param nMax: the number of G vectors in each direction to loop over. eg. 1 gives -1, 0, 1

    :param float cutoff: energy cutoff, in units of `2*(2\pi/L)^2`, defining the single-particle basis.  Only single-particle basis functions with a kinetic energy equal to or less than the cutoff are considered.

    :type sym: np.array
    :param sym: integer vector defining the wavevector, in units of `2\pi/L`, representing the desired symmetry.  Only Hartree products and determinants of this symmetry are returned.

    :returns: basis_fns where:

        basis_fns
            tuple of relevant :class:`PlaneWaveFn` objects, ie the single-particle basis set.

    '''

        # Single particle basis within the desired energy cutoff.
        #cutoff = cutoff*(2*np.pi/self.L)**2
        imax = int(np.ceil(np.sqrt(cutoff)))+1
        self.cutoff = cutoff
        self.imax = imax
        basis_fns = []
        for i in range(-imax, imax+1):
            for j in range(-imax, imax+1):
                for k in range(-imax, imax+1):
                    bfn = planewave.BasisFunc(i, j, k, self.L, 1)
                    if self.is_k_in_basis(bfn.k):
                        basis_fns.append(planewave.BasisFunc(i, j, k, self.L, 1))
                        basis_fns.append(planewave.BasisFunc(i, j, k, self.L, -1))
        # Sort in ascending order of kinetic energy.  Note that python's .sort()
        # (since 2.3) is guaranteed to be stable.
        basis_fns.sort()
        basis_fns = tuple(basis_fns)
        self.basis_fns = basis_fns

        self.init_basis_indices_map()

        return basis_fns


    def eval3BodyIntegrals(self, correlator = None,dtype=np.float64, sp=1):
        '''
            full 3-body integrals.
        '''
        algoName = "UEG.eval3BodyIntegrals"
        print_logging_info(algoName,level=0)
        startTime = time.time()

        world = ctf.comm()
        rank = world.rank()

        if self.basis_fns == None:
            raise BasisSetNotInitialized(algoName)
        if correlator is None:
            self.correlator = self.trunc
            print_logging_info("No correlator given.", level=1)
            print_logging_info("Using the default correlator: "\
                               +self.correlator.__name__, level=1)
        else:
            self.correlator = correlator
        if self.basis_indices_map is None:
            raise BasisFuncIndicesMapNotInitialised(algoName)

        print_logging_info(algoName)
        print_logging_info("Using TC method", level=1)
        print_logging_info("Using correlator:",correlator.__name__, level=1)
        print_logging_info("kCutoff in correlator:",self.kCutoff,level=1)

        nP = int(len(self.basis_fns)/2)
        tV_opqrst = ctf.tensor([nP,nP,nP,nP,nP,nP], dtype=dtype, sp=sp)
        indices = []
        values = []

        # due to the momentum conservation, only 5 indices are free.
        # implementation follow closely the get_lmat_ueg in NECI
        numKInEachDir = self.imax*2+1

        for o in range(nP):
            if (o) % world.np() == rank:
                print_logging_info("Elapsed time = {:.3f} s: calculating the {}-{} out of {} orbitals"\
                            .format(time.time()-startTime, o, \
                            o+world.np() if o+world.np() < nP else nP, nP),level=1)
                for r in range(nP):
                    kIntVec1 = self.basis_fns[2*r].k - self.basis_fns[2*o].k
                    for p in range(nP):
                        for s in range(nP):
                            kIntVec2 = self.basis_fns[2*p].k - self.basis_fns[2*s].k
                            for q in range(nP):
                                tIntVec = -kIntVec1 + kIntVec2 + self.basis_fns[2*q].k
                                locT = numKInEachDir**2 * (tIntVec[0] + self.imax) + \
                                        numKInEachDir * (tIntVec[1] + self.imax) +\
                                        tIntVec[2] + self.imax
                                if locT < len(self.basis_indices_map) and locT >= 0:
                                    t = int(self.basis_indices_map[locT])
                                    if t < 0:
                                        continue
                                else:
                                    continue

                                kVec1 = 2.0*np.pi/self.L*kIntVec1
                                kVec2 = 2.0*np.pi/self.L*kIntVec2
                                #kVec3 = -2.0*np.pi/self.L*(self.basis_fns[2*q].k-self.basis_fns[2*t].k)

                                w12 = self.correlator(kVec1.dot(kVec1)) \
                                        * self.correlator(kVec2.dot(kVec2))\
                                        * kVec1.dot(kVec2)
                                #w13 = self.correlator(kVec1.dot(kVec1))\
                                #        * self.correlator(kVec3.dot(kVec3))\
                                #        * kVec1.dot(kVec3)
                                #w23 = self.correlator(kVec2.dot(kVec2))\
                                #        * self.correlator(kVec3.dot(kVec3))\
                                #        * kVec2.dot(kVec3)
                                #w = (w12+w13+w23) / self.Omega**2
                                w = -(w12) / 2./ self.Omega**2
                                index = o*nP**5+p*nP**4+q*nP**3+r*nP**2+s*nP+t

                                values.append(w)
                                indices.append(index)
                                if index >= nP**6:
                                    raise("Index exceeds size of the tensor")

        tV_opqrst.write(indices,values)

        print_logging_info("{:.3f} s spent on ".format(time.time()-startTime)+algoName,\
                           level=1)

        return tV_opqrst



    def eval2BodyIntegrals(self, correlator = None, rpaApprox= False, \
            only2Body=False,onlyNonHermitian2Body=False,onlyHermitian2Body=False,\
            effective2Body= False,exchange1=False,exchange2=False, exchange3=False,\
            dtype=np.float64,sp=1):
        world = ctf.comm()
        algoName = "UEG.eval2BodyIntegrals"
        startTime = time.time()

        rank = world.rank()
        print_logging_info(algoName,level=0)

        if self.basis_fns == None:
            raise BasisSetNotInitialized(algoName)

        if correlator is not None:
            self.correlator = correlator
            print_logging_info("Using TC method", level=1)
            print_logging_info("Using correlator: ", correlator.__name__, level=1)
            if self.kCutoff is not None:
                print_logging_info("kCutoff in correlator = {:.8f}".format(self.kCutoff), level=1)
            if self.gamma is not None:
                print_logging_info("Gamma in correlator = {:.8f}".format(self.gamma), level=1)

            if only2Body:
                print_logging_info("Including only pure 2-body terms: ", only2Body, level=1)
            if rpaApprox:
                print_logging_info("Including only RPA approximation for 3-body: ",\
                               rpaApprox, level=1)
            if effective2Body:
                print_logging_info("Including effective 2-body terms from 3-body: "\
                               , effective2Body, level=1)
            if exchange1:
                print_logging_info("Using only 1. exchange type 2-body from 3-body: "\
                               , exchange1, level=1)
            if exchange2:
                print_logging_info("Using only 2. exchange type 2-body from 3-body: "\
                               , exchange2, level=1)

            if exchange3:
                print_logging_info("Using only 3. exchange type 2-body from 3-body: "\
                               , exchange3, level=1)
            # Just for testing, not used in production
            if onlyNonHermitian2Body:
                print_logging_info("Including only non-hermitian 2-body terms: "\
                               , onlyNonHermitian2Body, level=1)

            if onlyHermitian2Body:
                print_logging_info("Including only hermitian 2-body terms: "\
                               , onlyNonHermitian2Body, level=1)


        nP = int(len(self.basis_fns)/2)
        tV_pqrs = ctf.tensor([nP,nP,nP,nP], dtype=dtype, sp=sp)
        indices = []
        values = []


        numKInEachDir = self.imax*2+1

        for p in range(nP):
            if (p) % world.np() == rank:
                print_logging_info("Elapsed time = {:.3f} s: calculating the {}-{} out of {} orbitals"\
                                   .format(time.time()-startTime, p, \
                                    p+world.np() if p+world.np() < nP else nP, \
                                    nP), level=1)
                for r in range(nP):
                    dIntK = self.basis_fns[r*2].k-self.basis_fns[p*2].k
                    dKVec = self.basis_fns[r*2].kp-self.basis_fns[p*2].kp
                    uMat  = 0.
                    if correlator is not None:
                        uMat = self.sumNablaUSquare(dKVec)

                    for q in range(nP):
                        intKS = self.basis_fns[q*2].k - dIntK
                        #if self.is_k_in_basis(intKS):
                        locS = numKInEachDir**2 * (intKS[0] + self.imax) + \
                                numKInEachDir * (intKS[1] + self.imax) +\
                                intKS[2] + self.imax
                        if locS < len(self.basis_indices_map) and locS >= 0:
                            s = int(self.basis_indices_map[locS])
                            if s < 0:
                                continue
                        else:
                            continue

                        dkSquare = dKVec.dot(dKVec)
                        if correlator is None:
                            if np.abs(dkSquare) > 0.:
                                w = 4.*np.pi/dkSquare/self.Omega
                                indices.append(nP**3*p + nP**2*q + nP*r + s)
                                values.append(w)
                        elif rpaApprox:
                            # tc
                            if np.abs(dkSquare) > 0.:
                                rs_dk = self.basis_fns[r*2].kp-self.basis_fns[s*2].kp
                                w = 4.*np.pi/dkSquare \
                                        +  uMat\
                                        + dkSquare * correlator(dkSquare)\
                                        - (rs_dk.dot(dKVec)) * correlator(dkSquare) \
                                        - (self.nel-2)*dkSquare*correlator(dkSquare)**2/self.Omega
                                w = w / self.Omega
                            else:
                                #w = correlator(dkSquare, multiply_by_k_square=True) + uMat
                                w =  uMat / self.Omega
                                #w =  0.
                            indices.append(nP**3*p + nP**2*q + nP*r + s)
                            values.append(w)
                        elif only2Body:
                            if np.abs(dkSquare) > 0.:
                                rs_dk = self.basis_fns[r*2].kp-self.basis_fns[s*2].kp
                                w = 4.*np.pi/dkSquare \
                                        +  uMat\
                                        + dkSquare * correlator(dkSquare)\
                                        - (rs_dk.dot(dKVec)) * correlator(dkSquare)
                                        #- (self.nel - 2)*dkSquare*correlator(dkSquare)**2/self.Omega
                                w = w / self.Omega
                            else:
                                #w = correlator(dkSquare, multiply_by_k_square=True) + uMat
                                w =  uMat / self.Omega
                            indices.append(nP**3*p + nP**2*q + nP*r + s)
                            values.append(w)
                        elif onlyHermitian2Body:
                            if np.abs(dkSquare) > 0.:
                                rs_dk = self.basis_fns[r*2].kp-self.basis_fns[s*2].kp
                                w = 4.*np.pi/dkSquare \
                                        +  uMat\
                                        + dkSquare * correlator(dkSquare)
                                        #- (self.nel - 2)*dkSquare*correlator(dkSquare)**2/self.Omega
                                w = w / self.Omega
                            else:
                                #w = correlator(dkSquare, multiply_by_k_square=True) + uMat
                                w =  uMat / self.Omega
                            indices.append(nP**3*p + nP**2*q + nP*r + s)
                            values.append(w)
                        elif onlyNonHermitian2Body:
                            w = 0.
                            if np.abs(dkSquare) > 0.:
                                rs_dk = self.basis_fns[r*2].kp-self.basis_fns[s*2].kp
                                w = 4.*np.pi/dkSquare \
                                        - (rs_dk.dot(dKVec)) * correlator(dkSquare)
                                        #- (self.nel - 2)*dkSquare*correlator(dkSquare)**2/self.Omega
                                w = w / self.Omega
                            #else:
                            #    #w = correlator(dkSquare, multiply_by_k_square=True) + uMat
                            #    w =  uMat / self.Omega

                                # \sum_k' (k-k')k'u(k-k')u(k')

                            indices.append(nP**3*p + nP**2*q + nP*r + s)
                            values.append(w)
                        elif effective2Body:
                            if np.abs(dkSquare) > 0.:
                                w = - (self.nel)*dkSquare*correlator(dkSquare)**2/self.Omega\
                                    + 2.*self.contractExchange3Body(self.basis_fns[2*r].kp, dKVec)\
                                    - 2.*self.contractExchange3Body(self.basis_fns[2*p].kp, dKVec)\
                                    + 2.*self.contractP_KWithQ(self.basis_fns[2*r].kp, dKVec)
                                w = w / self.Omega
                            else:
                                #w = correlator(dkSquare, multiply_by_k_square=True) + uMat
                                w =  (2.*self.contractP_KWithQ(self.basis_fns[2*r].kp, dKVec))
                                w = w / self.Omega
                            indices.append(nP**3*p + nP**2*q + nP*r + s)
                            values.append(w)
                        # exchange 1-3 are for test purpose only.
                        elif exchange1:
                            if np.abs(dkSquare) > 0.:
                                w = + 2.*self.contractExchange3Body(self.basis_fns[2*r].kp, dKVec)
                                w = w / self.Omega
                            else:
                                #w = correlator(dkSquare, multiply_by_k_square=True) + uMat
                                w =  0.
                            indices.append(nP**3*p + nP**2*q + nP*r + s)
                            values.append(w)
                        elif exchange2:
                            if np.abs(dkSquare) > 0.:
                                w = - 2.*self.contractExchange3Body(self.basis_fns[2*p].kp, dKVec)
                                w = w / self.Omega
                            else:
                                #w = correlator(dkSquare, multiply_by_k_square=True) + uMat
                                w =  0.
                            indices.append(nP**3*p + nP**2*q + nP*r + s)
                            values.append(w)
                        elif exchange3:
                            if np.abs(dkSquare) > 0.:
                                w = + 2.*self.contractP_KWithQ(self.basis_fns[2*r].kp, dKVec)
                                w = w / self.Omega
                            else:
                                #w = correlator(dkSquare, multiply_by_k_square=True) + uMat
                                w = + 2.*self.contractP_KWithQ(self.basis_fns[2*r].kp, dKVec)
                                w = w / self.Omega
                            indices.append(nP**3*p + nP**2*q + nP*r + s)
                            values.append(w)
                            #tV_pqrs[p,q,r,s] = 4.*np.pi/dkSquare/sys.Omega
        tV_pqrs.write(indices,values)

        if effective2Body:
            # symmetrize the integral with respect to electron 1 and 2
            tV_sym_pqrs = ctf.tensor(tV_pqrs.shape, sp=tV_pqrs.sp)
            tV_sym_pqrs.i("pqrs") << 0.5*(tV_pqrs.i("pqrs")+tV_pqrs.i("qpsr"))
            del tV_pqrs
            tV_pqrs = tV_sym_pqrs

        print_logging_info("{:.3f} s spent on ".format(time.time()-startTime)+algoName,\
                           level=1)
        return tV_pqrs


    def contractExchange3Body(self, pVec, kVec):
        '''
        pVec and kVec should be a single vector
        kVec is the momentum transfer
        '''
        pPrim = np.array([self.basis_fns[i*2].kp for i in range(int(self.nel/2))])
        pVec = pVec-pPrim
        kVecSquare = np.einsum("i,i->", kVec, kVec)
        pVecSquare = np.einsum("ni,ni->n", pVec, pVec)
        pVecDotKVec = np.einsum("ni,i->n", pVec, kVec)
        result = pVecDotKVec * self.correlator(kVecSquare) \
                * self.correlator(pVecSquare)
        result = np.einsum("n->", result) / self.Omega
        #result = result * (self.nel-2.)/self.nel

        return result


    def contractP_KWithQ(self, pVec, kVec):

        pPrim = np.array([self.basis_fns[i*2].kp for i in range(int(self.nel/2))])
        vec1 = pVec - kVec - pPrim
        vec2 = pVec-pPrim

        dotProduct = np.einsum("ni,ni->n", vec1, vec2)
        vec1Square = np.einsum("ni,ni->n", vec1, vec1)
        vec2Square = np.einsum("ni,ni->n", vec2, vec2)
        result = dotProduct * self.correlator(vec1Square) * self.correlator(vec2Square)

        result = np.einsum("n->",result) / self.Omega
        #result = result * (self.nel-2.)/self.nel

        return result


    def contract3BodyIntegralsTo2Body(self, integrals):
        # factor 2 for the spin degeneracy
        fac = 2
        RPA2Body = fac*ctf.einsum("opqrsq->oprs", integrals)
        return RPA2Body

    def sumNablaUSquare(self, k, cutoff=30):
        # need to test convergence of this cutoff
        if self.kPrime is None:
            self.kPrime = np.array([[i,j,k] for i in range(-cutoff,cutoff+1) \
                    for j in range(-cutoff,cutoff+1) for k in range(-cutoff,cutoff+1)])
        k1 = 2*np.pi*self.kPrime/self.L
        k2 = k - k1

        k1Square = np.einsum("ni,ni->n", k1,k1)
        k2Square = np.einsum("ni,ni->n", k2,k2)
        k1DotK2 = np.einsum("ni,ni->n", k1,k2)
        result = k1DotK2 * self.correlator(k1Square) * self.correlator(k2Square)
        result = np.einsum("n->", result) / self.Omega

        return result


    def triple_contractions_in_3_body(self):
        """
        This function computes the triply contracted 3-body interactions.
        Return: a scalar (float) which should be added to the total energy
        """
        algo_name = "UEG.triple_contractions_in_3_body"
        print_logging_info(algo_name, level=1)

        p = np.array([self.basis_fns[i*2].kp for i in range(int(self.nel/2))])
        q = np.array([self.basis_fns[i*2].kp for i in range(int(self.nel/2))])
        tp_pi = ctf.astensor(p)
        tq_qi = ctf.astensor(q)
        tp_q_pqi = ctf.tensor([len(p),len(q),3])
        tp_q_pqi.i("pqi") << tp_pi.i("pi")-tq_qi.i("qi")
        tp_qSquare_pq = ctf.einsum("pqi,pqi->pq", tp_q_pqi, tp_q_pqi)

        p_qSquare = tp_qSquare_pq.to_nparray()
        up_q = self.correlator(p_qSquare)

        dirE = up_q**2*p_qSquare
        # factor 2 comes from sum over spins
        dirE = sum(sum(dirE)) * self.nel/2/self.Omega**2 * 2

        # exchange type
        tUp_q_pq = ctf.astensor(up_q)
        tp_oDotp_q = ctf.einsum("poi,pqi->pqo", tp_q_pqi, tp_q_pqi)

        UpqUpo = ctf.einsum("pq,po->pqo",tUp_q_pq,tUp_q_pq)
        # factor 2 from sum over spin, another factor of 2 from mirror symmetry
        excE = -2*2*ctf.einsum("pqo,pqo->", tp_oDotp_q, UpqUpo)/2./self.Omega**2
        result = dirE+excE
        print_logging_info("Direct E = {:.8f}".format(dirE), level=2)
        print_logging_info("Exchange E = {:.8f}".format(excE), level=2)

        return result

    def double_contractions_in_3_body(self):
        """
            This function makes two contractions in the 3-body integrals,
            resulting in one body energies, which should be added to the
            original one-particle energies from the HF theory

            return: a numpy array of size equal to the number of plane waves
        """
        algo_name = "UEG.double_contractions_in_3_body"
        print_logging_info(algo_name, level=1)
        # some constants

        num_o = int(self.nel/2)
        num_v = int(len(self.basis_fns)/2)-num_o
        num_p = num_o + num_v

        # initialize the one_particle_energies
        one_particle_energies = np.zeros(num_p)
        e_perl = np.zeros(num_p)

        # generate p vectors
        k_vec_p = np.array([self.basis_fns[i*2].kp for i in range(int(len(self.basis_fns)/2))])
        k_vec_i = np.array([self.basis_fns[i*2].kp for i in range(int(self.nel/2))])

        # the perl shape diagram
        for orb_p in range(num_p):
            k_vec_p_minus_i = k_vec_p[orb_p] - k_vec_i
            k_vec_p_minus_i_square = np.einsum("ij, ij-> i", k_vec_p_minus_i, k_vec_p_minus_i)
            e_perl[orb_p] = np.sum(self.correlator(k_vec_p_minus_i_square)**2\
                                   *k_vec_p_minus_i_square)

        e_perl = 2.0*self.nel/self.Omega**2/2 * e_perl

        one_particle_energies += e_perl

        # wave diagram
        e_wave = np.zeros(num_p)
        t_diff_vec_pi_pij = ctf.tensor([num_p, num_o, 3])
        t_diff_vec_pi_pij.i("pij") << ctf.astensor(k_vec_p).i("pj")-ctf.astensor(k_vec_i).i("ij")
        t_diff_vec_pi_square_pi = ctf.einsum("pij,pij -> pi", t_diff_vec_pi_pij, \
                                             t_diff_vec_pi_pij)
        t_diff_pi_dot_diff_pj_pij = ctf.einsum("pik, pjk -> pij", t_diff_vec_pi_pij, \
                                               t_diff_vec_pi_pij)
        diff_pi_square = t_diff_vec_pi_square_pi.to_nparray()
        u_diff_pi = self.correlator(diff_pi_square)
        t_u_diff_pi_multiply_u_diff_pj_pij = ctf.einsum("pi,pj->pij", u_diff_pi, u_diff_pi)
        e_wave = ctf.einsum("pij,pij->p", t_diff_pi_dot_diff_pj_pij, \
                            t_u_diff_pi_multiply_u_diff_pj_pij)

        e_wave = -e_wave * 2 / self.Omega**2/2

        one_particle_energies += e_wave.to_nparray()



        # shield diagram, which is independent of vector p. So initialize as
        # ones

        e_shield = np.ones(num_p)
        t_diff_vec_ij_ijk = ctf.tensor([num_o, num_o, 3])
        t_diff_vec_ij_ijk.i("ijk") << ctf.astensor(k_vec_i).i("ik")-ctf.astensor(k_vec_i).i("jk")
        t_diff_vec_ij_square_ij = ctf.einsum("ijk,ijk -> ij", t_diff_vec_ij_ijk, t_diff_vec_ij_ijk)
        diff_ij_square = t_diff_vec_ij_square_ij.to_nparray()
        u_diff_ij = self.correlator(diff_ij_square)
        u_diff_ij_square = u_diff_ij**2
        e_shield = e_shield * ctf.einsum("ij,ij->", u_diff_ij_square, diff_ij_square)
        # bug, missing a factor of 2 from spin degree of freedom
        e_shield =  2*e_shield / 2 / self.Omega**2

        one_particle_energies += e_shield

        # frog diagram (there are two types which turn out to be the same, so
        # a factor of 4 will be multiplied in the end)
        e_frog = np.zeros(num_p)
        # Using -(p-i) as vector (i-p), pay attention to the exchange of p and
        # i indices
        t_diff_ij_dot_diff_ip_ijp = ctf.einsum("ijk, pik -> ijp", \
                                               t_diff_vec_ij_ijk, \
                                               -t_diff_vec_pi_pij)

        t_u_diff_ij_multiply_u_diff_ip_ijp = ctf.einsum("ij,pi->ijp", u_diff_ij,\
                                                        u_diff_pi)
        e_frog = ctf.einsum("ijp, ijp->p", t_diff_ij_dot_diff_ip_ijp, \
                            t_u_diff_ij_multiply_u_diff_ip_ijp)

        e_frog = -e_frog * 4 / self.Omega**2/2

        one_particle_energies += e_frog.to_nparray()



        return one_particle_energies

    # collection of correlators, should them be collected into a class?
    # each correlator has some default parameters that are dependent on
    # the system and they are specific to UEG, so they should be part of
    # the UEG class.


    def yukawa(self, kSquare, multiply_by_k_square=False):
        '''
        The G=0 terms need more consideration
        '''
        rho = self.nel / self.Omega
        gamma_0 = np.sqrt(4.* np.pi * rho)
        if self.gamma is None:
            #gamma = np.sqrt(4.*(3.*rho/np.pi)**(1/3.))
            gamma = gamma_0
        else:
            gamma = self.gamma * gamma_0
        # has to be - and divided by gamm to satisfy the cusp condition
        a = -4.*np.pi*2
        if self.kCutoff is not None:
            kCutoffSquare = self.kCutoff * ((2*np.pi/self.L)**2)
            kCutoffDenom = kCutoffSquare*(kCutoffSquare + gamma**2)
        else:
            kCutoffDenom = 1e-12
        if not multiply_by_k_square:
            b = kSquare*(kSquare+gamma**2)
            result = np.divide(a , b, out = np.zeros_like(b), where=np.abs(b)>kCutoffDenom)
        else:
            if kSquare > kCutoffSquare:
                result = a/(kSquare+gamma**2)
            else:
                result = 0.

        return result


    def trunc(self, kSquare, multiply_by_k_square=False):
        '''
        The G=0 terms need more consideration
        '''
        if self.kCutoff is None:
            self.kCutoff = int(np.ceil(np.sqrt(self.cutoff)))

        if self.gamma is None:
            self.gamma = 1.0

        kCutoffSquare = (self.kCutoff * 2*np.pi/self.L)**2

        if not isinstance(kSquare, np.ndarray):
            if kSquare <= kCutoffSquare*(1+0.00001):
                kSquare = 0.
        else:
            kSquare[kSquare <= kCutoffSquare*(1+0.00001)] = 0.
        result = np.divide(-4.*np.pi, kSquare**2, out = np.zeros_like(kSquare),\
                where=(kSquare > 1e-12))
        return result*self.gamma

    def gaskell_modified(self, kSquare, multiply_by_k_square=False):
        '''
        input: G^2, will be scaled by k_fermi as beta^2=G^2/k_f^2
        output: \mu/beta^2, beta<2; 4\mu/beta^4, beta>2
        '''
        # define the parameter mu in gaskell correlator
        # this calculation will be done multipule times, it is not optimal to
        # recalculate it everytime. After refactoring all the correlators into
        # classes, this problem can be solved by using it as parameter of the
        # gaskall correlator class and only initialize it once. For now I will
        # keep it here.
        if self.kCutoff is not None:
            kCutoffSquare =  (self.kCutoff*(2*np.pi/self.L))**2
        else:
            kCutoffSquare = 2
        mu = np.pi
        #k_fermi.dot(k_fermi)

        if not isinstance(kSquare, np.ndarray):
            result = 0.
            if kSquare < kCutoffSquare and kSquare > 1e-12:
                #result = 4*mu/kSquare
                result = 0.
            else:
                result = 4*mu/kSquare**2
        else:
            result = np.divide(0.*mu, kSquare, out = np.zeros_like(kSquare),\
                where=(kSquare > 1e-12))
            result[kSquare>=kCutoffSquare] = 0.
            result += np.divide(4*mu, kSquare**2, out = np.zeros_like(kSquare),\
                where=(kSquare >= kCutoffSquare))
        # there should be an overall - sign
        return -result

    def gaskell(self, kSquare, multiply_by_k_square=False):
        '''
        input: G^2, will be scaled by k_fermi as beta^2=G^2/k_f^2
        output: \mu/beta^2, beta<2; 4\mu/beta^4, beta>2
        '''
        # define the parameter mu in gaskell correlator
        # this calculation will be done multipule times, it is not optimal to
        # recalculate it everytime. After refactoring all the correlators into
        # classes, this problem can be solved by using it as parameter of the
        # gaskall correlator class and only initialize it once. For now I will
        # keep it here.
        mu = (1/(3*np.pi)*(4./(9*np.pi))**(1./3)*self.rs)**(1./2)
        k_fermi = self.basis_fns[int(self.nel/2)*2].kp
        int_k_fermi = self.basis_fns[int(self.nel/2)*2].k
        beta_square= kSquare / (k_fermi.dot(k_fermi))


        if self.kCutoff is not None:
            kCutoffSquare =  self.kCutoff**2/int_k_fermi.dot(int_k_fermi)
        else:
            kCutoffSquare = 4
        #k_fermi.dot(k_fermi)

        if not isinstance(kSquare, np.ndarray):
            result = 0.
            if beta_square < kCutoffSquare and beta_square > 1e-12:
                result = mu/beta_square
            else:
                result = 4*mu/beta_square**2
        else:
            result = np.divide(mu, beta_square, out = np.zeros_like(kSquare),\
                where=(beta_square > 1e-12))
            result[beta_square>kCutoffSquare] = 0.
            result += np.divide(4*mu, beta_square**2, out = np.zeros_like(kSquare),\
                where=(beta_square >= kCutoffSquare))
                #where=(beta_square > 1e-12 and beta_square <= kCutoffSquare))
        # there should be an overall - sign
        return -result

    def smooth(self, kSquare, multiply_by_k_square=False):
        '''
        The G=0 terms need more consideration
        '''
        if self.kCutoff is None:
            self.kCutoff = int(np.ceil(np.sqrt(self.cutoff)))

        if self.gamma is None:
            self.gamma = 0.01

        kCutoffSquare = (self.kCutoff * 2*np.pi/self.L)**2
        #kCutoffSquare = self.kCutoff**2
        #if (kSquare <= ktc_cutoffSquare):
        #    result = 0.
        #else:
        #    result = - 12.566370614359173 / kSquare/kSquare

        #result = np.divide(-4.*np.pi*applyErf(kSquare,kCutoffSquare), kSquare**2, \
        #        out = np.zeros_like(kSquare), where=kSquare>1e-12)
        kc = np.sqrt(kCutoffSquare)
        k = np.sqrt(kSquare)
        result = np.divide(-4.*np.pi*(1.+special.erf((k-kc)/(kc*self.gamma)))/2., kSquare**2, \
                out = np.zeros_like(kSquare), where=kSquare>(kc*self.gamma)**2)
        return result

    def coulomb(self, kSquare, multiply_by_k_square=False):
        '''
        The G=0 terms need more consideration
        '''
        if self.gamma is None:
            gamma = 1.
        else:
            gamma = self.gamma
        result = np.divide(-4.*np.pi*gamma, kSquare, out = np.zeros_like(kSquare), where=kSquare>1e-12)
        return result

    def stg(self,kSquare,multiply_by_k_square=False):
        if self.gamma is None:
            rho = self.nel / self.Omega
            gamma = np.sqrt(4.* np.pi * rho)
            #gamma = np.sqrt(4.*(3.*rho/np.pi)**(1/3.))
        else:
            gamma = self.gamma
        a = -4.*np.pi/gamma
        if self.kCutoff is not None:
            kCutoffSquare = self.kCutoff * ((2*np.pi/self.L)**2)
            kCutoffDenom = (kCutoffSquare + gamma**2)**2
        else:
            kCutoffDenom = 1e-12
        if not multiply_by_k_square:
            b = (kSquare+gamma**2)**2
            result = np.divide(a , b, out = np.zeros_like(b), where=np.abs(b)>kCutoffDenom)

        return result


    # interface to CC4S, for test purpose only, not essential

    def calcGamma(self, overlap_basis, nP):
        '''
        FTOD : Fourier Transformed Overlap Density
        return: C^p_q({\bf G}) = \int\mathrm d{\bf r} \phi^*_p({\bf r}\phi_q({\bf r})e^{i{\bf G\cdot r}}
        '''
        algoName = "UEG.calcGamma"
        if self.basis_fns == None:
            raise BasisSetNotInitialized(algoName)

        nG = int(len(overlap_basis)/2)
        gamma_pqG = np.zeros((nP,nP,nG))

        for p in range(0,nP,1):
            for q in range(0,nP,1):
                for g in range(0,nG,1):
                    if ((basis[2*p].k-basis[2*q].k) == overlap_basis[2*g].k).all():
                        GSquare = overlap_basis[2*g].kp.dot(overlap_basis[2*g].kp)
                        if np.abs(GSquare) > 1e-12 :
                            gamma_pqG[p,q,g] = np.sqrt(4.*np.pi/GSquare/self.Omega)
        return gamma_pqG

