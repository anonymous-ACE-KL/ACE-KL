
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from ae_tf import AutoEncoder_tf
from sklearn import preprocessing
import pandas as pd
import sys
from scipy.stats import spearmanr, describe
from ace import ace_tabular, ace_tf_regression
from ace import lime_tabular
import csv
import cPickle as cpkl

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_magnitude(most_inds, acekl_mag, ace_mag, lime_mag, direc_mag, value_mag, examed_dim=12, filename='example'):

    fig, ax = plt.subplots()
    indices = np.arange(examed_dim)*4
    bar_width = 1
    opacity = 0.8

    rects0 = plt.bar(indices - 1*bar_width, acekl_mag, bar_width,
                     alpha=opacity,
                     color='c',
                     label='ACE_KL')

    rects1 = plt.bar(indices, ace_mag, bar_width,
                     alpha=opacity,
                     color='b',
                     label='ACE')

    rects2 = plt.bar(indices + 1*bar_width, lime_mag, bar_width,
                     alpha=opacity,
                     color='y',
                     label='LIME')

    '''
    rects3 = plt.bar(indices + 1*bar_width, direc_mag, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Direc')
    '''

    # print scale_value.shape
    '''
    rects4 = plt.bar(indices + 2*bar_width, value_mag, bar_width,
                     alpha=opacity,
                     color='r',
                     label='scale')
    '''

    plt.xlabel('Variable Index')
    plt.ylabel('Contribution')
    # plt.title('Scores by person')
    plt.xticks(indices, np.asarray(most_inds))
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename+'-newmag.pdf', format = 'pdf')

def plot_kl(acekl_mag, ace_mag, lime_mag, direc_mag, value_mag, filename):

    kl = [np.sum(value_mag*np.log(value_mag*1.0/acekl_mag)),
            np.sum(value_mag*np.log(value_mag*1.0/ace_mag)),
            np.sum(value_mag*np.log(value_mag*1.0/lime_mag))]
            # np.sum(value_mag*np.log(value_mag*1.0/direc_mag))

    fig, ax = plt.subplots()
    bar_width = 0.8
    opacity = 0.8

    rects = plt.bar(np.arange(1, 4), kl, bar_width,
                     alpha=opacity)

    rects[0].set_color('c')
    rects[1].set_color('b')
    rects[2].set_color('y')
    # rects[3].set_color('g')

    plt.xlabel('Method')
    plt.ylabel('KL Divergence')
    # plt.title('Scores by person')
    plt.xticks(np.arange(1, 5), ['ACE_KL', 'ACE', 'LIME'])

    c_patch = mpatches.Patch(color='c', label='ACE_KL')
    blue_patch = mpatches.Patch(color='blue', label='ACE')
    yellow_patch = mpatches.Patch(color='yellow', label='LIME')
    # green_patch = mpatches.Patch(color='green', label='AE')

    plt.legend(handles=[c_patch, blue_patch, yellow_patch])

    plt.tight_layout()
    plt.savefig(filename+'-newkl.pdf', format = 'pdf')


def present_anomaly(examed_example, most_inds, var2namedic, filename):
    with open(filename+"-present.txt", "w") as f:
        for ind in most_inds:
            value = examed_example[ind]
            varname = var2namedic[ind]
            f.write(varname+" : "+str(value)+"\n")


def main():
    epochs = 100
    batch_size = 400
    input_dim = 122
    examed_dim = 12
    rng = np.random.RandomState(12345)

    # var2namedic = dict([(i, k[0]) for i, k in enumerate(varmeanls)])


    '''
    Read the data
    '''

    Data_X = np.load(sys.argv[1])
    Data_y = np.load(sys.argv[2])

    try:
        with tf.device("/gpu:0"):
            print "Using gpu!"
            ace_model = ace_tf_regression.AceRegression(input_dim)
    except:
        with tf.device("/cpu:0"):
            print "Using cpu!"
            ace_model = ace_tf_regression.AceRegression(input_dim)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        for test_id, X, y in zip(range(10), Data_X, Data_y):
            sess.run(init)


            feature_names = [str(x) for x in range(input_dim)]
            examed_example = X[0]

            """
            ace_kl
            """
            MyAceKlexp = ace_tf_regression.AceKlTabularExplainer(sess, ace_model, examed_example, input_dim)
            print 'Training ACE_KL...'
            acekl_exp = MyAceKlexp.explain_instance(X,
                                                    y,
                                                    num_features=input_dim)
            print 'Trained ACE_KL.'
            acekl_res = sorted(acekl_exp.as_map(), key=lambda x: x[0])


            """
            ace
            """
            ace_explainer = ace_tabular.AceTabularExplainer(examed_example, input_dim)
            print 'Training ACE...'
            ace_exp = ace_explainer.explain_instance(X,
                                                    y,
                                                    num_features=input_dim)
            print 'Trained ACE.'
            ace_res = sorted(ace_exp.as_map(), key=lambda x: x[0])


            """
            lime
            """
            lime_explainer = lime_tabular.LimeTabularExplainer(examed_example, input_dim)
            print 'Training LIME...'
            lime_exp = lime_explainer.explain_instance(X,
                                                        y,
                                                        num_features=input_dim)
            print 'Trained LIME.'
            lime_res = sorted(lime_exp.as_map(), key=lambda x: x[0])


            most_scaled_res = sorted([(i, np.abs(v)) for i, v in enumerate(examed_example)], key=lambda x: x[1], reverse=True)[:examed_dim]

            # print most_scaled_res

            most_inds = [k for k, v in most_scaled_res]

            # most_direc_res = [direc_res[k] for k in most_inds]

            # print len(lime_res), len(most_inds)
            most_lime_res = [lime_res[k] for k in most_inds]

            most_ace_res = [ace_res[k] for k in most_inds]

            most_acekl_res = [acekl_res[k] for k in most_inds]

            acekl_array = np.asarray([np.abs(most_acekl_res[i][1]) for i in range(examed_dim)])
            # print lime_array
            acekl_mag = acekl_array/np.sum(acekl_array)

            lime_array = np.asarray([np.abs(most_lime_res[i][1]) for i in range(examed_dim)])
            # print lime_array
            # import pdb; pdb.set_trace()
            lime_mag = lime_array/np.sum(lime_array)

            ace_array = np.asarray([np.abs(most_ace_res[i][1]) for i in range(examed_dim)])
            # print ace_array
            ace_mag = ace_array/np.sum(ace_array)

            # direc_array = np.asarray([np.abs(most_direc_res[i][1]) for i in range(examed_dim)])
            # print direc_array
            # direc_mag = direc_array/np.sum(direc_array)
            direc_mag = None

            value_array = np.asarray([np.abs(most_scaled_res[i][1]) for i in range(examed_dim)])
            # print value_array
            value_mag = value_array/np.sum(value_array)

            # print most_direc_res, most_lime_res, most_ace_res
            filename = str(test_id)
            plot_magnitude(most_inds, acekl_mag, ace_mag, lime_mag, direc_mag, value_mag,
                            examed_dim, filename)

            plot_kl(acekl_mag, ace_mag, lime_mag, direc_mag, value_mag, filename)

            pdfrm = pd.DataFrame(X[:, np.asarray(most_inds)])

            present_anomaly(examed_example, most_inds, var2namedic, filename)

if __name__ == '__main__':
    main()

