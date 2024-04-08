#!/usr/bin/python3

from sys import argv
from collections import defaultdict

class integrals:
    def __init__(self,fcidump,tcdump=None):
        self.fcidump = fcidump
        self.tcdump = tcdump
        print("FCIDUMP =",self.fcidump,"| TCDUMP =", self.tcdump)
        self.integrals = {} 
        self.__read_fcidump__()
        self.__energy_2body__()
        self.__fill_missing__()
        if self.tcdump:
            self.__read_tcdump__()
            self.__energy_3body__()
            self.__mean_field_three_body__()
            self.__energy_2body__()

    def __read_fcidump__(self):
        with open(self.fcidump,'r') as f:
            lines = f.readlines()
        self.norb = int(lines[0].split("NORB=")[-1].split(',')[0])
        self.nelec = int(lines[0].split("NELEC=")[-1].split(',')[0])
        self.nocc = int(self.nelec/2) #assume closed shell case
        #self.nocc = self.nelec
        for i,line in enumerate(lines):
            if "&End".upper() in line.upper() or "/" in line:
                integral_start = i+1
                break
        for line in lines[integral_start:]:
            words = line.split()
            value = float(words[0])
            indices = tuple([int(word) for word in words[1:]])
            self.integrals[indices] = value

    def __energy_2body__(self):
        energy = [0. for i in range(3)]
        enuc_indx = (0,0,0,0)
        energy[0] = self.integrals[enuc_indx]
        #print(self.integrals[enuc_indx])
        for i in range(1,self.nocc+1):
            #print(2.,(i,i),self.__get_1body__(i,i))
            energy[1] += 2.*self.__get_1body__(i,i)
            for j in range(1,self.nocc+1):
                #print(2.,(i,i,j,j),self.__get_2body__(i,i,j,j))
                #print(-1.,(i,j,j,i),self.__get_2body__(i,j,j,i))
                energy[2] += 2.*self.__get_2body__(i,i,j,j)\
                               -self.__get_2body__(i,j,j,i)
        print('Hartree-Fock Energy (0body):',energy[0])
        print('Hartree-Fock Energy (1body):',energy[1])
        print('Hartree-Fock Energy (2body):',energy[2])
        print('Hartree-Fock Energy (sum):',sum(energy))
        print(50*'-')

    def __energy_3body__(self):
        energy = [0. for i in range(4)]
        enuc_indx = (0,0,0,0)
        energy[0] = self.integrals[enuc_indx]
        #print(self.integrals[enuc_indx])
        for i in range(1,self.nocc+1):
            #print(2.,(i,i),self.__get_1body__(i,i))
            energy[1] += 2.*self.__get_1body__(i,i)
            for j in range(1,self.nocc+1):
                #print(2.,(i,i,j,j),self.__get_2body__(i,i,j,j))
                #print(-1.,(i,j,j,i),self.__get_2body__(i,j,j,i))
                energy[2] += 2.*self.__get_2body__(i,i,j,j)\
                               -self.__get_2body__(i,j,j,i)
                for k in range(1,self.nocc+1):
                    #print(4./3.,(i,i,j,j,k,k),self.__get_3body__(i,i,j,j,k,k))
                    #print(2./3.,(i,j,j,k,k,i),self.__get_3body__(i,j,j,k,k,i))
                    #print(-2.,(i,j,j,i,k,k),self.__get_3body__(i,j,j,i,k,k))
                    energy[3] += 4./3.*self.__get_3body__(i,i,j,j,k,k)\
                            + 2./3.*self.__get_3body__(i,j,j,k,k,i)\
                            -    2.*self.__get_3body__(i,j,j,i,k,k)
        print('Hartree-Fock Energy (0body):',energy[0])
        print('Hartree-Fock Energy (1body):',energy[1])
        print('Hartree-Fock Energy (2body):',energy[2])
        print('Hartree-Fock Energy (3body):',energy[3])
        print('Hartree-Fock Energy (sum(0,1,2)):',sum(energy[:3]))
        print('Hartree-Fock Energy (sum):',sum(energy))
        print(50*'-')

    def __fill_missing__(self):
        for p in range(1,self.norb+1):
            for q in range(1,self.norb+1):
                oneel_indx = (p,q,0,0)
                if oneel_indx not in self.integrals:
                    self.integrals[oneel_indx] = self.__get_1body__(p,q)
                for r in range(1,self.norb+1):
                    for s in range(1,self.norb+1):
                        twoel_indx = (p,q,r,s)
                        if twoel_indx not in self.integrals:
                            self.integrals[twoel_indx] = self.__get_2body__(p,q,r,s)

    def __read_tcdump__(self):
        with open(self.tcdump) as f:
            lines = f.readlines()
            for line in lines[1:]:
                words = line.split()
                #value = -3.*float(words[0]) #don't ask ...
                value = 3.*float(words[0]) #don't ask ...
                indices = tuple([int(word) for word in [words[4],words[1],words[5],words[2],words[6],words[3]]])
                #print(indices,value)
                self.integrals[indices] = value

    #def __read_tcdump__(self):
    #    with open(self.tcdump) as f:
    #        lines = f.readlines()
    #        nintegrals = int((len(lines) - 1)/2)
    #        for i in range(nintegrals):
    #            indx = 1 + 2*i
    #            words = (lines[indx]+lines[indx+1]).split()
    #            value = float(words[0])
    #            #indices = tuple([int(word) for word in words[1:]])
    #            indices = tuple([int(word) for word in [words[1],words[4],words[2],words[5],words[3],words[6]]])
    #            print(indices)
    #            #self.integrals_3body[indices] = -3.* value #don't ask ...
    #            self.integrals_3body[indices] = 3.* value #don't ask ...

    def __mean_field_three_body__(self):
        #print(self.nocc)
        enuc_indx = (0,0,0,0)
        for i in range(1,self.nocc+1):
            for j in range(1,self.nocc+1):
                for k in range(1,self.nocc+1):
                    #print(self.integrals[enuc_indx])
                    #print(self.__get_3body__(i,i,j,j,k,k),self.__get_3body__(i,j,j,k,k,i),self.__get_3body__(i,j,j,i,k,k))
                    self.integrals[enuc_indx] +=\
                            + 4./3. * self.__get_3body__(i,i,j,j,k,k)\
                            + 2./3. * self.__get_3body__(i,j,j,k,k,i)\
                            - 2.    * self.__get_3body__(i,j,j,i,k,k)

        for p in range(1,self.norb+1):
            for q in range(1,self.norb+1):
                oneel_indx = (p,q,0,0)
                for i in range(1,self.nocc+1):
                    for j in range(1,self.nocc+1):
                        #print(p,q,self.integrals[oneel_indx])
                        #print(p,q,self.__get_3body__(p,q,i,i,j,j),self.__get_3body__(p,q,i,j,j,i),self.__get_3body__(p,i,i,q,j,j),self.__get_3body__(p,i,j,q,i,j))
                        self.integrals[oneel_indx] -=\
                                + 2. * self.__get_3body__(p,q,i,i,j,j)\
                                -      self.__get_3body__(p,q,i,j,j,i)\
                                - 2. * self.__get_3body__(p,i,i,q,j,j)\
                                +      self.__get_3body__(p,i,j,q,i,j)
        
        for p in range(1,self.norb+1):
            for q in range(1,self.norb+1):
                for r in range(1,self.norb+1):
                    for s in range(1,self.norb+1):
                        twoel_indx = (p,q,r,s)
                        for i in range(1,self.nocc+1):
                            #print(p,q,r,s,self.integrals[twoel_indx])
                            #print(p,q,r,s,self.__get_3body__(p,q,r,s,i,i),self.__get_3body__(p,q,r,i,i,s),self.__get_3body__(r,s,p,i,i,q))
                            self.integrals[twoel_indx] +=\
                                    + 2. * self.__get_3body__(p,q,r,s,i,i)\
                                    -      self.__get_3body__(p,q,r,i,i,s)\
                                    -      self.__get_3body__(r,s,p,i,i,q)

    def __get_3body__(self,p,q,r,s,t,u):
        permutations = (
                (p,q,r,s,t,u),
                (p,q,t,u,r,s),
                (r,s,p,q,t,u),
                (r,s,t,u,p,q),
                (t,u,p,q,r,s),
                (t,u,r,s,p,q),
                (q,p,r,s,t,u),
                (q,p,t,u,r,s),
                (r,s,q,p,t,u),
                (r,s,t,u,q,p),
                (t,u,q,p,r,s),
                (t,u,r,s,q,p),
                (p,q,s,r,t,u),
                (p,q,t,u,s,r),
                (s,r,p,q,t,u),
                (s,r,t,u,p,q),
                (t,u,p,q,s,r),
                (t,u,s,r,p,q),
                (q,p,s,r,t,u),
                (q,p,t,u,s,r),
                (s,r,q,p,t,u),
                (s,r,t,u,q,p),
                (t,u,q,p,s,r),
                (t,u,s,r,q,p),
                (p,q,r,s,u,t),
                (p,q,u,t,r,s),
                (r,s,p,q,u,t),
                (r,s,u,t,p,q),
                (u,t,p,q,r,s),
                (u,t,r,s,p,q),
                (q,p,r,s,u,t),
                (q,p,u,t,r,s),
                (r,s,q,p,u,t),
                (r,s,u,t,q,p),
                (u,t,q,p,r,s),
                (u,t,r,s,q,p),
                (p,q,s,r,u,t),
                (p,q,u,t,s,r),
                (s,r,p,q,u,t),
                (s,r,u,t,p,q),
                (u,t,p,q,s,r),
                (u,t,s,r,p,q),
                (q,p,s,r,u,t),
                (q,p,u,t,s,r),
                (s,r,q,p,u,t),
                (s,r,u,t,q,p),
                (u,t,q,p,s,r),
                (u,t,s,r,q,p),
                )
        for p in permutations:
            if p in self.integrals:
                return self.integrals[p]
        else:
            return 0.0

    def __get_2body__(self,p,q,r,s):
        permutations = ((p,q,r,s),(r,s,p,q))
        for p in permutations:
            if p in self.integrals:
                return self.integrals[p]
        else:
            return 0.0

    def __get_1body__(self,p,q):
        indx = (p,q,0,0)
        if indx in self.integrals:
            return self.integrals[indx]
        else:
            return 0.0

    def __repr__(self):
        epsilon = 10**-8
        outlines = []
        outlines.append(str(self.norb))
        for indx,value in sorted(self.integrals.items()):
            if indx[-1] == 0 and abs(value) > epsilon:
                outlines.append(' : '.join([str(indx),str(value)]))

        for indx,value in sorted(self.integrals.items()):
            if len(indx) == 4 and indx[-1] != 0 and abs(value) > epsilon:
                outlines.append(' : '.join([str(indx),str(value)]))

        for indx,value in sorted(self.integrals.items()):
            if len(indx) == 6 and abs(value) > epsilon:
                outlines.append(' : '.join([str(indx),str(value)]))
        return '\n'.join(outlines)

def main(fcidump_old,tcdump_old):
    #with open('FC_py','w') as f:
    #    f.write(str(integrals(fcidump)))
    with open('FC_old_py','w') as f:
        f.write(str(integrals(fcidump_old)))
    with open('FC_TC_py','w') as f:
        f.write(str(integrals(fcidump_old,tcdump_old)))

if __name__ == '__main__':
    #fcidump = argv[1]
    fcidump_old = argv[1]
    tcdump_old = argv[2]
    main(fcidump_old,tcdump_old)
