#!/usr/bin/python3

import os
import sys
import numpy as np
import spglib as spg
import symmetrize as symm
#from ase import build
eps = sys.float_info.epsilon*10


class Structure:
    def __init__(self,fileName=None):
        """
        all the coordinates are scaled, if true distance is needed, always 
        remenber to multiply by the latticeConstant
        """
        self.cellVecs = np.zeros((3,3))
        self.latticeConstant = 1.0
        self.numAtom = 1
        self.posAtom = np.zeros((self.numAtom,3))
        self.fileName = fileName
        self.fileHeader = "header"
        self.typeCor = "D"
        self.atomSpec = "H"
        self.spaceGroup= None
        self.spgCell = None


        if fileName is not None:
            self.readFromFile(fileName)
        self.spgCell = self.convert2SpgCell()

    def convert2SpgCell(self):
        self.spgCell = (self.cellVecs.T*self.latticeConstant, self.posAtom, np.ones(self.numAtom))
        return self.spgCell

    def getSpacegroup(self, symprec=0.01):
        self.spaceGroup = spg.get_spacegroup(self.spgCell, symprec=symprec)
        return self.spaceGroup

    def getPrimitiveCell(self, symprec=0.01):
        pc = spg.find_primitive(self.spgCell, symprec=symprec)
        return pc

    def direct2Cart(self, coor):
        cart = np.zeros(np.shape(coor))
        #print("Unit cellVecs Vectors=\n", self.cellVecs)
        #for i in range(np.shape(cart)[0]):
        #    cart[i,:] = self.cellVecs.dot(coor[i,:])
        cart = self.cellVecs.dot(coor.T).T
        return cart

    def cart2Direct(self, coor):
        direct = np.zeros(np.shape(coor))
        #print("Unit cellVecs Vectors=\n", self.cellVecs)
        #print("Inverted Unit cellVecs Vectors=\n", np.linalg.inv(self.cellVecs))
        #for i in range(np.shape(direct)[0]):
        #    direct[:] = np.linalg.inv(self.cellVecs).dot(coor[:])
        direct = np.linalg.inv(self.cellVecs).dot(coor.T).T
        return direct

    def getDistance(self,i,j):
        if self.typeCor.lower() == "d" or self.typeCor.lower() == "direct":
            posI = self.direct2Cart(self.posAtom[i])
            posJ = self.direct2Cart(self.posAtom[j])
        else:
            posI = self.posAtom[i]
            posJ = self.posAtom[j]
        return np.linalg.norm(posI-posJ)*self.latticeConstant

    def getDistance(self,posI,posJ):
        """
        inputs two positions in cartesion coordinates and return the distance    
        """
        return np.linalg.norm(posI-posJ)*self.latticeConstant

    def findNNTable(self):
        NNTable = np.zeros((self.numAtom, self.numAtom))
        if self.typeCor.lower() == "d" or self.typeCor.lower() == "direct":
            coorInCart = self.direct2Cart(self.posAtom)
        else:
            coorInCart = self.posAtom
        zeroVec = np.zeros(3)
        for i in range(self.numAtom):
            for j in range(i+1,self.numAtom):
                ijMirrorDistances = []
                for shift1 in [zeroVec, self.cellVecs.T[0], -self.cellVecs.T[0]]:
                    for shift2 in [zeroVec, self.cellVecs.T[1], -self.cellVecs.T[1]]:
                        for shift3 in [zeroVec, self.cellVecs.T[2], -self.cellVecs.T[2]]:
                            totalShift = shift1+shift2+shift3
                            ijMirrorDistances.append(self.getDistance(coorInCart[i], coorInCart[j]+totalShift))
                miniDistance = np.min(np.array(ijMirrorDistances))
                NNTable[i,j] = miniDistance
                NNTable[j,i] = miniDistance

        return NNTable

    def readFromFile(self, fileName=None):
        # Read structure information from file. Update all parameters.
        skiprows=0
        with open(fileName, 'r') as f:
            self.fileHeader = next(f)
            skiprows=skiprows+1
            #print("Structure file header=", self.fileHeader)
            self.latticeConstant = float(next(f))
            skiprows=skiprows+1
            #print("Lattice constant=", self.latticeConstant)
            self.cellVecs[:,0] = np.array(next(f).split())
            skiprows=skiprows+1
            #print("Unit cellVecs vector a=",self.cellVecs[:,0])
            self.cellVecs[:,1] = np.array(next(f).split())
            skiprows=skiprows+1
            #print("Unit cellVecs vector b=",self.cellVecs[:,1])
            self.cellVecs[:,2] = np.array(next(f).split())
            skiprows=skiprows+1
            #print("Unit cellVecs vector c=",self.cellVecs[:,2])
            self.atomSpec = next(f)
            skiprows=skiprows+1
            try:
                self.atomSpec=int(self.atomSpec)
                self.numAtom = self.atomSpec
                self.atomSpec="H"
            except:
                self.atomSpec = self.atomSpec.strip()[0]
                self.numAtom = int(next(f))
                skiprows=skiprows+1
            #print("Number of atoms read from "+fileName+"=",self.numAtom)
            self.typeCor = next(f).strip()[0]
            skiprows=skiprows+1
            #print("Coordinates type=",self.typeCor)
            #self.posAtom = np.loadtxt(fileName, maxrows=self.numAtom, skiprows=skiprows)
            self.posAtom = np.loadtxt(fileName, skiprows=skiprows)
            #print("Coordinates of atoms=\n",self.posAtom)
        f.close()
        return

    def write2File(self, fileName=None):
        # write new structure to file
        with open("StructureHistory.dat",'a') as f:
            f.write(self.fileHeader)
            f.write(str(self.latticeConstant)+"\n")
        f.close()
        with open("StructureHistory.dat",'ab') as f:
            np.savetxt(f, self.cellVecs[:,:].T)
        f.close()
        with open("StructureHistory.dat",'a') as f:
            f.write(str(self.atomSpec)+"\n")
            f.write(str(self.numAtom)+"\n")
            f.write(str(self.typeCor)+"\n")
        f.close()
        with open("StructureHistory.dat",'ab') as f:
            np.savetxt(f, self.posAtom)
        f.close()
        if fileName is not None:
            with open(fileName,'w') as f:
                f.write(self.fileHeader)
                f.write(str(self.latticeConstant)+"\n")
            f.close()
            with open(fileName,'ab') as f:
                np.savetxt(f, self.cellVecs[:,:].T)
            f.close()
            with open(fileName,'a') as f:
                f.write(str(self.atomSpec)+"\n")
                f.write(str(self.numAtom)+"\n")
                f.write(str(self.typeCor)+"\n")
            f.close()
            with open(fileName,'ab') as f:
                np.savetxt(f, self.posAtom)
            f.close()

        return


class Optimizer:
    def __init__(self,structure, threshhold=1e-3, symprec = 0.01, timestep=0.01):
        self.HFForces = np.zeros((1,3))
        self.MP2Forces = np.zeros((1,3))
        self.totalForces = np.zeros((1,3))
        self.structure = structure
        self.spgCell = None
        self.posAtom    = np.zeros((1,3))
        self.numAtom    = structure.numAtom
        self.timeStep = timestep
        self.threshhold = threshhold
        self.structureUpdated=0
        self.primitiveCellMap=None
        self.isCellPrimitive = False
        self.primCellStruct = None
        self.primSpgCell = None
        self.transMatrix = None
        self.symprec = symprec
        #self.checkStructure()

    def convert2SpgCell(self, structure):
        spgCell = (structure.cellVecs.T*structure.latticeConstant, structure.posAtom, np.ones(structure.numAtom))
        return spgCell

    #def checkStructure(self):
    #    """ 
    #    check on the supplied structure
    #    """
    #    # construct the tuple for spg lib
    #    self.spgCell = self.convert2SpgCell(self.structure)
    #    dataSet = spg.get_symmetry_dataset(self.spgCell)
    #    primitive_cell = spg.standardize_cell(self.spgCell, to_primitive=False, no_idealize=True)
    #    print(primitive_cell[0].T)
    #    print(len(primitive_cell[1]))
    #    if dataSet is not None:
    #        #prim = spg.find_primitive(self.spgCell)
    #        # get map to primitive cell
    #        # are the two primitive cells the same?
    #        print("cp cells lattice")
    #        print(np.linalg.det(self.spgCell[0].T))

    #        #print("prim cells lattice")
    #        #print(np.linalg.det(dataSet["primitive_lattice"].T))
    #        #print("Map to Prim")
    #        #print(self.map2Pc)
    #        #self.map2Pc = list(set(self.map2Pc))
    #        #print("list of Map to Prim")
    #        #print(self.map2Pc)
    #        #self.isCellPrimitive = True
    #        self.primCellStruct = Structure()
    #        self.primCellStruct.cellVecs = dataSet["primitive_lattice"].T
    #        ## to get the position of atoms in prim, we need
    #        ## to the transformation matrix between the original 
    #        ## and the prim cells.
    #        #self.transMatrix = self.primCellStruct.cellVecs.dot(np.linalg.inv(self.spgCell[0].T))
    #        self.transMatrix = np.array([[1,1,1],[0,1,1],[0,-3,0]])
    #        #self.transMatrix = np.array([[1,0,0],[0,1,0],[0,0,1]])
    #        print("transMatrix")
    #        print(self.transMatrix)
    #        print("det(transMatrix)")
    #        print(np.linalg.det(self.transMatrix))
    #        print("Prim lattice vecs")
    #        
    #        print(dataSet["primitive_lattice"].T)
    #        print("Transformed sp lattice to prim lattice")
    #        transPrimCellVecs = (np.linalg.inv(self.transMatrix).dot(self.spgCell[0]))
    #        transPrimCellVecs[np.abs(transPrimCellVecs)<eps]=0
    #        print(transPrimCellVecs)

    #        self.primCellStruct.posAtom = (self.structure.posAtom[:,:]).dot((self.transMatrix))
    #        #self.primCellStruct.posAtom = self.transMatrix.dot(self.structure.posAtom[:,:].T).T
    #        print(self.primCellStruct.posAtom)
    #        # see if after transformation there are correct number of 
    #        # atoms inside of the primitive cell.
    #        atom_pos = []
    #        for i in (self.primCellStruct.posAtom):
    #            if all(i>0.) and all(i < 1.) :
    #                atom_pos.append(i)
    #        print(atom_pos)
    #        print(len(atom_pos))
    #        self.primCellStruct.latticeConstant = 1.0
    #        #self.primCellStruct.numAtom = len(self.map2Pc)
    #        self.primCellStruct.fileHeader = self.structure.fileHeader
    #        self.primCellStruct.typeCor = self.structure.typeCor
    #        self.primCellStruct.atomSpec = self.structure.atomSpec
    #        self.primSpgCell = self.convert2SpgCell(self.primCellStruct)
    #        #print("prim cell pos")
    #        #print(self.primCellStruct.posAtom)
    #        #print("writing PPOSCAR to file...")
    #        self.primCellStruct.write2File("PPOSCAR.test")
            


    def getForces(self):
        self.getHFForces()
        self.getMP2Forces()
        if len(self.HFForces) != len(self.MP2Forces):
            with open("structOp.log",'a+') as f:
                print("HF forces and Mp2 forces have different dimensions! \
                        will not add together to get total forces!")
            f.close
        else:
            self.totalForces=self.HFForces+self.MP2Forces
        return self.totalForces
    
    
    def getHFForces(self, fileName="HFForces.dat"):
        if os.path.isfile(fileName):
            data = np.loadtxt(fileName)
            self.HFForces = data[:,3:]
            self.posAtom = data[:,0:3]
            self.numAtom = len(data[:,0])
        else:
            self.HFForces = np.zeros((self.numAtom,3))
        return self.HFForces


    
    def getMP2Forces(self, fileName="Mp2Forces.dat"):
        if os.path.isfile(fileName):
            with open(fileName,'r') as f:
                header= next(f).split()
                #print("header=",header)
                self.numAtom = int(header[3])
            f.close()
            self.MP2Forces = np.array(np.loadtxt(fileName, skiprows=2))
            #print("number of atoms from Mp2Forces=",self.numAtom)
            self.MP2Forces = self.MP2Forces.reshape((self.numAtom,3))
        else:
            self.MP2Forces = np.zeros((self.numAtom,3))
        return self.MP2Forces



    def project2PrimitiveCell(self, forces, map2PC=None):
        """
         this means we are using forces from a supercell
         to update a primitive cell
         get the ion indices of atoms from PC inside of SC
         map2Pc: its a map of atoms in the supercell to primitive cell
        """
        if map2PC is None:
            map2Pc=np.loadtxt("ionIndices.dat").astype(int)-1
        forces=forces[map2Pc[:,1],:]
        return forces


    def symmetrizeForces(self, forces, spgCell=None):
        """
         this function refer to ASE.spacegroup.symmetrize. 
         in the future this script will be merged into ASE. But for 
         now just to make it work in this script.
         cell: is a tuple contains the lattice vectors, atom positions and atom spec
                if not supplied then will use the default structure within the class
         forces: should have the same 0st dimension as the atom positions
        """

        
        if spgCell is None:
            spgCell = (self.structure.cellVecs.T*self.structure.latticeConstant, self.structure.posAtom, np.ones(self.numAtom))
        if len(forces[:,0]) != len(spgCell[1]):
            print("Dimensions of atoms and forces don't match!")
            sys.exit(1)
        dataset = spg.get_symmetry_dataset(spgCell, symprec=self.symprec)
        with open("structOp.log","a+") as log:
            log.write(spg.get_spacegroup(spgCell, symprec=self.symprec)+"\n")
        log.close()
        fixSymm = symm.FixSymmetry(spgCell, verbose=False)
        fixSymm.adjust_forces(spgCell, forces)

        return forces

    def updateStructure(self,  HFForces=None, MP2Forces=None, symmtrize=True, inPC=False):
        """
        forces: if external forces are supllied, then use them
        inPC: flag to tell if update the corresponding primitive cell or in
              the supercell directly.
        """
        if HFForces is not None: 
            self.HFForces = HFForces
        else:
            self.getHFForces()
            if symmtrize:
                self.symmetrizeForces(self.HFForces)

        if MP2Forces is not None:
            self.MP2Forces = MP2Forces
        else:
            self.getMP2Forces()
        self.totalForces = self.HFForces + self.MP2Forces
        #if symmtrize:
        #    if len(self.HFForces) != len(self.MP2Forces) and inPC:
        #        self.symmetrizeForces(self.MP2Forces)
        #        self.project2PrimitiveCell(self.Mp2Forces)
        #    self.symmetrizeForces(self.totalForces)
        maxForce=(self.totalForces[:,0]**2+self.totalForces[:,1]**2+self.totalForces[:,2]**2).max()**(1/2.)
        maxHFForce=(self.HFForces[:,0]**2+self.HFForces[:,1]**2+self.HFForces[:,2]**2).max()**(1/2.)
        maxMP2Force=(self.MP2Forces[:,0]**2+self.MP2Forces[:,1]**2+self.MP2Forces[:,2]**2).max()**(1/2.)
        with open("force.log","a+") as log:
            log.write(str(maxHFForce)+" "+str(maxMP2Force)+" "+str(maxForce)+"\n")
        log.close()

        np.savetxt("symmForces.dat", self.totalForces)
        np.savetxt("symmHFForces.dat", self.HFForces)
        np.savetxt("symmMP2Forces.dat", self.MP2Forces)

        if maxForce > self.threshhold:    
            #print("Maximum force on atom=", maxForce, r'eV/$\AA$')
            #print("Updating structure...")
            # for now simple gradient descent.
            self.structure.posAtom = self.structure.posAtom + self.structure.cart2Direct(
                    self.totalForces*self.timeStep/self.structure.latticeConstant
                    )
            self.structureUpdated=1
        else:
            #print("Forces are smaller than supplied threshhold, will not update structure!")
            self.structureUpdated=0
        #self.structure.write2File()
        return self.structure

def main():
    if len(sys.argv) > 1:
        thresh = float(sys.argv[1])
        if len(sys.argv) > 2:
            timestep=float(sys.argv[2])
        else:
            timestep = 0.01
    else:
        thresh = 5e-2
        timestep = 0.01
    pc=Structure()
    pc.readFromFile("PPOSCAR")


    sc=Structure()
    sc.readFromFile("POSCAR")
    optSc = Optimizer(sc, thresh, symprec=0.01, timestep=timestep)
    optPc = Optimizer(pc, thresh, symprec=0.01, timestep=timestep)
    spgPc = optPc.convert2SpgCell(pc)
    spgSc = optSc.convert2SpgCell(sc)
    Mp2forces = optSc.getMP2Forces()
    # symmetrize Mp2 forces using supercell symmetries
    Mp2forces = optSc.symmetrizeForces(Mp2forces,spgSc)
    # map the forces from supercell to primitive cell atoms
    Mp2forces = optSc.project2PrimitiveCell(Mp2forces)
    np.savetxt("nonSymmMp2Froces.dat",Mp2forces)
    # symmetrize forces using the pc symmetries
    Mp2forces = optSc.symmetrizeForces(Mp2forces, spgPc)
    transfromMatrix=(sc.cellVecs.T*sc.latticeConstant).dot(np.linalg.inv(pc.cellVecs.T*pc.latticeConstant))
    transfromMatrix[np.abs(transfromMatrix)<eps]=0
    #print(transfromMatrix)
    #print("det",np.linalg.det(transfromMatrix))
    #print("sc original")
    #print(sc.cellVecs.T)
    #print("sc")
    #print(transfromMatrix.dot(pc.cellVecs.T*pc.latticeConstant)/sc.latticeConstant)
    transfromMatrix=np.rint(transfromMatrix)
    np.savetxt("transMat.dat",transfromMatrix)
    pc = optPc.updateStructure(MP2Forces=Mp2forces)
    pc.write2File()


    print(optPc.structureUpdated)

if __name__ == '__main__':
    main()
