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
from lime import lime_tabular
import csv
import cPickle as cpkl


def boxplot_func(name, data):
    fig, ax = plt.subplots()

    ax.boxplot(data, notch=True)
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Variance')
    ax.set_title(name)
    plt.tight_layout()
    plt.savefig(name+'.png', format = 'png')
    plt.close(fig)

def main():
    train_pklfilename = sys.argv[1]
    test_pklfilename = sys.argv[2]

    varmeanls = [('n_std_src_ports', 1), ('n_nonstd_src_ports', 0), ('n_dst_ip', 1), ('avg_std_src_ports_per_dst_ip', 1), ('avg_nonstd_src_ports_per_dst_ip', 0), ('max_n_src_ports_by_dst_ip', 1), ('min_n_src_ports_by_dst_ip', 1), ('n_std_dst_ports', 0), ('n_nonstd_dst_ports', 1), ('n_dst_ip_std_dst_ports', 0), ('n_dst_ip_nonstd_dst_ports', 1), ('n_flows_out', 1), ('total_duration_out', 15.676), ('max_duration_out', 15.676), ('min_duration_out', 15.676), ('avg_duration_per_dst_ip', 15.676), ('max_duration_per_dst_ip', 15.676), ('min_duration_per_dst_ip', 15.676), ('n_packets_out', 17), ('max_n_packets_out', 17), ('min_n_packets_out', 17), ('avg_n_packets_per_dst_ip', 17), ('max_n_packets_per_dst_ip', 17), ('min_n_packets_per_dst_ip', 17), ('n_bytes_out', 19697), ('max_n_bytes_out', 19697), ('min_n_bytes_out', 19697), ('avg_n_bytes_per_dst_ip', 19697), ('max_n_bytes_per_dst_ip', 19697), ('min_n_bytes_per_dst_ip', 19697), ('n_std_dst_ports', 1), ('n_nonstd_dst_ports', 0), ('n_src_ip', 1), ('avg_std_dst_ports_per_src_ip', 1), ('avg_nonstd_dst_ports_per_src_ip', 0), ('max_n_dst_ports_by_src_ip', 1), ('min_n_dst_ports_by_src_ip', 1), ('n_std_src_ports', 0), ('n_nonstd_src_ports', 1), ('n_src_ip_std_src_ports', 0), ('n_src_ip_nonstd_src_ports', 1), ('n_flows_in', 2), ('total_duration_in', 118.621), ('max_duration_in', 102.246), ('min_duration_in', 16.375), ('avg_duration_per_src_ip', 118.621), ('max_duration_per_src_ip', 118.621), ('min_duration_per_src_ip', 118.621), ('n_packets_in', 18), ('max_n_packets_in', 12), ('min_n_packets_in', 6), ('avg_n_packets_per_src_ip', 18), ('max_n_packets_per_src_ip', 18), ('min_n_packets_per_src_ip', 18), ('n_bytes_in', 480), ('max_n_bytes_in', 360), ('min_n_bytes_in', 120), ('avg_n_bytes_per_src_ip', 480), ('max_n_bytes_per_src_ip', 480), ('min_n_bytes_per_src_ip', 480), ('tos_out_1', 0), ('tos_out_2', 0), ('tos_out_3', 0), ('tos_out_4', 0), ('tos_out_5', 0), ('tos_out_6', 0), ('tos_out_7', 0), ('tos_out_8', 0), ('protos_out_1', 1), ('protos_out_2', 0), ('protos_out_3', 0), ('flags_out_1', 0), ('flags_out_2', 0), ('flags_out_3', 0), ('flags_out_4', 1), ('flags_out_5', 1), ('flags_out_6', 0), ('flags_out_7', 1), ('flags_out_8', 1), ('tos_in_1', 0), ('tos_in_2', 0), ('tos_in_3', 0), ('tos_in_4', 0), ('tos_in_5', 0), ('tos_in_6', 0), ('tos_in_7', 0), ('tos_in_8', 0), ('protos_in_1', 1), ('protos_in_2', 0), ('protos_in_3', 0), ('flags_in_1', 0), ('flags_in_2', 0), ('flags_in_3', 0), ('flags_in_4', 1), ('flags_in_5', 1), ('flags_in_6', 0), ('flags_in_7', 1), ('flags_in_8', 1)]

    var2namedic = dict([(i, k[0]) for i, k in enumerate(varmeanls)])

    var2namedic.update(dict([(98, 'top1_out'), (99, 'top2_out'), (100, 'top3_out'), (101, 'top4_out'), (102, 'top5_out'), (103, 'top1_in'), (104, 'top2_in'), (105, 'top3_in'), (106, 'top4_in'), (107, 'top5_in')]))

    with open(train_pklfilename, "r") as f:
        normaldataX, _ = cpkl.load(f)
    with open(test_pklfilename, "r") as f:
        anomalydataX = cpkl.load(f)

    print normaldataX.shape

    for ind in range(108):
        name = str(ind)+"_"+var2namedic[ind]
        data = normaldataX[:,ind]
        boxplot_func(name, data)


if __name__ == '__main__':
    main()
