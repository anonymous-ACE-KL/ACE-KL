# lime_ae.py

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import lime
from lime import lime_tabular
from ae_tf import AutoEncoder_tf
from sklearn import preprocessing
import pandas as pd
import sys
from scipy.stats import spearmanr, describe

def plot_ranking(filename, tuples1, tuples2):
    fig, ax = plt.subplots()
    ax.scatter(tuples1, tuples2, c = range(12), marker='s')
    for i in range(12):
        ax.annotate(i, (tuples1[i]+0.25, tuples2[i]+0.25))
    plt.xlim((-1,12))
    plt.ylim((-1,12))
    # plt.legend(loc = 0)
    plt.xticks(np.arange(-1, 13))
    plt.yticks(np.arange(-1, 13))
    plt.xlabel("Lime Variable Ranking")
    plt.ylabel("AE Variable Ranking")
    rho, p = spearmanr(tuples1, tuples2)
    plt.title('rho: ' + str(rho) + ', p: '+str(p))
    plt.savefig(filename+'_pt.png', format = 'png')

def plot_magnitude(lime_res, direc_res, scale_value, filename):
    lime_mag = np.asarray([np.absolute(lime_res[i][1]) for i in range(12)])/np.sum([np.absolute(lime_res[i][1]) for i in range(12)])
    direc_mag = np.asarray([np.sqrt(direc_res[i][1]) for i in range(12)])
    # value_mag = np.asarray([np.absolute(scale_value[i]) for i in range(12)])/np.sum([np.absolute(scale_value[i]) for i in range(12)])

    fig, ax = plt.subplots()
    index = np.arange(12)
    bar_width = 0.33
    opacity = 0.8

    rects1 = plt.bar(index, lime_mag, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Lime')

    rects2 = plt.bar(index + bar_width, direc_mag, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Direc')

    rects3 = plt.bar(index + 2*bar_width, np.abs(scale_value), bar_width,
                 alpha=opacity,
                 color='r',
                 label='scale')

    plt.xlabel('Variable Index')
    plt.ylabel('Contribution')
    # plt.title('Scores by person')
    plt.xticks(index + 1.5*bar_width, index)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename+'_mag.png', format = 'png')


def gen_syndata(rng, input_dim):
    mean = np.asarray([2, 0, 53, 185, 27, 15172, 195, 29, 16166, 13, 2, 793])
    # cov = np.identity(100)
    rawdata = rng.poisson(mean, size = (4000, input_dim)) +1
    # dataX = rng.multivariate_normal(mean, cov, 5000)
    return rawdata

def perturb(rng, input_dim, data, perturb_ind=1):
    # mean = np.arange(input_dim-1, -1, -1)
    mean = np.zeros(input_dim)
    mean[perturb_ind] = mean[perturb_ind]+10000
    # print mean
    # cov = np.eye(100)*5.0
    noise = rng.poisson(mean, size = (1000, input_dim)).astype("float32")
    # noise = rng.multivariate_normal(mean, cov, 1000)
    positive_data = data + noise
    # print data[10]
    # print positive_data[10]
    return positive_data

def main():
    epochs = 100
    batch_size = 400
    input_dim = 12
    hidden_dim = 6
    rng = np.random.RandomState(12345)
    csv_in_file_name = sys.argv[1]
    test_id = int(sys.argv[2])
    perturb_ind = int(sys.argv[3])

    try:
        with tf.device("/gpu:0"):
            print "Using gpu!"
            ae = AutoEncoder_tf(rng, input_dim, hidden_dim)
    except:
        with tf.device("/cpu:0"):
            print "Using cpu!"
            ae = AutoEncoder_tf(rng, input_dim, hidden_dim)

    # Train
    # min_max_scaler = preprocessing.MinMaxScaler()
    # rawdataX = gen_syndata(rng, input_dim)

    rawdataX = pd.read_csv(csv_in_file_name, header=None).as_matrix()
    # '''
    # print 'before', rawdataX[10]
    raw_testX_positive = perturb(rng, input_dim, rawdataX[3000:], perturb_ind)
    # dataX = preprocessing.scale(np.concatenate((rawdataX, raw_testX_positive), axis = 0))
    mean_std_scaler = preprocessing.StandardScaler().fit(rawdataX.astype(np.float))
    trainX = mean_std_scaler.transform(rawdataX.astype(np.float))
    # dataX = preprocessing.normalize(rawdataX, norm='l2')
    # trainX = dataX[:4000]
    # testX_positive = dataX[4000:]
    testX_positive = mean_std_scaler.transform(raw_testX_positive)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print 'Training AutoEncoder...'
        for epoch in range(epochs):
            rng.shuffle(trainX)
            for batch_ind in range(10):
                batch_xs = trainX[batch_ind*batch_size: (batch_ind+1)*batch_size]
                # print batch_xs[0]
                train_loss = ae.train(batch_xs, sess)

            # print 'epoch, loss = {}: {}'.format(epoch, train_loss)
        print 'Trained AutoEncoder.'
        # print 'loss (train) = ', ae.predict([trainX[0]])

        feature_names = [str(x) for x in range(input_dim)]
        explainer = lime_tabular.LimeTabularExplainer(trainX, feature_names = feature_names, class_names=['Normal'], verbose=True)

        # test_id = 8
        # examed_example = trainX[3000+test_id]
        # examed_example = testX_positive[test_id]
        examed_example = rawdataX[test_id] + np.asarray([100, 8, 100, 172, 30, 30000, 200, 31, 1000, 14, 0, 800])


        scaled_examed_example = mean_std_scaler.transform(examed_example.reshape(1, -1).astype(np.float)).flatten()
        print scaled_examed_example

        print 'Training LIME...'
        exp = explainer.explain_instance(scaled_examed_example, ae.calas, labels=[0], num_features=12)
        print 'Trained LIME.'

        # print exp.as_map()[0]

        lime_res = sorted(exp.as_map()[0], key=lambda x: x[0])
        sorted_lime_res = sorted(lime_res, key=lambda x: np.absolute(x[1]), reverse = True)
        print "lime", sorted_lime_res

        lime_ind_ord = [ele[0] for ele in sorted_lime_res]
        # print lime_ind_ord
        lime_to_figure = [lime_ind_ord.index(u) for u in range(12)]
        # print lime_to_figure

        # print scaled_examed_example
        # print ae.predict(np.asarray([scaled_examed_example]))[0]
        direc_res = [(i, v) for i, v in enumerate((scaled_examed_example-ae.predict(np.asarray([scaled_examed_example]))[0])**2)]
        sorted_direc_res = sorted(direc_res, key=lambda x: x[1], reverse = True)
        print "AE", sorted_direc_res

        direc_ind_ord = [ele[0] for ele in sorted_direc_res]
        # print direc_ind_ord
        direc_to_figure = [direc_ind_ord.index(u) for u in range(12)]
        # print direc_to_figure

    # plot_ranking(str(test_id), lime_to_figure, direc_to_figure)

    plot_magnitude(lime_res, direc_res, scaled_examed_example, str(test_id))


if __name__ == '__main__':
    main()

