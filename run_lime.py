import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import preprocessing
import pandas as pd
import sys
from scipy.stats import spearmanr, describe
from ace import ace_explainer_pt
import csv
import cPickle as cpkl
import pdb

def main():

    """
    load data
    """
    X = np.load(sys.argv[1])
    X = np.where(X == 0, -1, 1).astype(np.float32)
    y = np.load(sys.argv[2])
    y = -np.log(y).astype(np.float32)


    examed_example = X[0]
    input_dim = len(examed_example)

    # pdb.set_trace()

    """
    lime
    """
    lime_explner = ace_explainer_pt.AceExplainer(examed_example, input_dim, is_lime=True)
    print 'Training Lime...'
    lime_exp = lime_explner.explain_instance(X,
                                            y)
    print 'Trained Lime.'
    print lime_exp.as_map()
    lime_res = sorted(lime_exp.as_map(), key=lambda x: x[0])
    print lime_res

    # pdb.set_trace()

if __name__ == '__main__':
    main()
