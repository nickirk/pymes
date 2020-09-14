from string import ascii_lowercase
import numpy as np

def write2Cc4sTensor(tensor, dim, fileName, dtype="r"):
    # This function write a tensor to the format that 
    # cc4s can read in. Only text file format is available for
    # now. It should only be used for tests in small systems.
    # first deal with the header
    f = open(fileName+".dat", "w")
    dimString=""
    for i in dim:
        dimString=dimString+" "+str(i)
    f.write(fileName+" "+dimString+"\n")
    indices=""
    for i in ascii_lowercase[8:8+dim[0]]:
        indices=indices+i
    f.write(indices+" \n")
    f.close()

    # now write the data to file
    f = open(fileName+".dat", "a")
    tensor=tensor.flatten("C")

    if dtype == "c":
        np.savetxt(f, tensor, fmt="(%.18e,%.18e)")
    else:
        np.savetxt(f, tensor, fmt="%.18e")
    f.close()
    return
