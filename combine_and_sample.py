import sys
import numpy as np
import pdb

with open(sys.argv[1]) as f:
    lines_x = f.readlines()

with open(sys.argv[2]) as h:
    lines_y = h.readlines()


X, Y = [], []
for x_line, y_line in zip(lines_x, lines_y):
    x = x_line.replace('\n', '').strip().split(' ')
    # print x
    x = [float(element) for element in x]
    X.append(x)
    y = np.log(float(y_line.replace('\n', '')))
    Y.append(y)


X = np.asarray(X)
Y = np.asarray(Y)
sort_ind = np.argsort(Y).astype(np.int)

X = X[sort_ind]
Y = -Y[sort_ind]

# pdb.set_trace()
###
# sampling around the true x, we use the most anomaly examples
###

sample_size = 4999
p = 0.05

###
# 1st sample
###

flip = np.random.binomial(1, p, (sample_size,len(X[0])))
orig = np.repeat(X[0][np.newaxis,:], sample_size, axis = 0)
x = np.logical_xor(orig, flip).astype(np.int)
x = np.concatenate((X[0][np.newaxis,:], x), axis = 0)

np.savetxt("bad0.dat", x, fmt="%d", delimiter=" ")
# pdb.set_trace()

###
# 2nd sample
###

flip = np.random.binomial(1, p, (sample_size,len(X[1])))
orig = np.repeat(X[1][np.newaxis,:], sample_size, axis = 0)
x = np.logical_xor(orig, flip).astype(np.int)
x = np.concatenate((X[1][np.newaxis,:], x), axis = 0)

np.savetxt("bad1.dat", x, fmt="%d", delimiter=" ")

###
# 3rd sample
###

flip = np.random.binomial(1, p, (sample_size,len(X[2])))
orig = np.repeat(X[2][np.newaxis,:], sample_size, axis = 0)
x = np.logical_xor(orig, flip).astype(np.int)
x = np.concatenate((X[2][np.newaxis,:], x), axis = 0)

np.savetxt("bad2.dat", x, fmt="%d", delimiter=" ")

###
# 4th sample
###

flip = np.random.binomial(1, p, (sample_size,len(X[3])))
orig = np.repeat(X[3][np.newaxis,:], sample_size, axis = 0)
x = np.logical_xor(orig, flip).astype(np.int)
x = np.concatenate((X[3][np.newaxis,:], x), axis = 0)

np.savetxt("bad3.dat", x, fmt="%d", delimiter=" ")
