import numpy as np
import sys
import pdb

def main():
    filenamex = sys.argv[1]
    filenamey = sys.argv[2]
    X = np.loadtxt(filenamex, dtype = np.float, delimiter = " ")
    Y = np.loadtxt(filenamey, dtype = np.float, delimiter = " ")
    np.save("x"+filenamex.split(".")[0], X)
    np.save("y"+filenamey.split(".")[0], Y)
    # pdb.set_trace()


if __name__ == '__main__':
    main()
