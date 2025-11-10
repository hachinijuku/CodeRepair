#! /usr/bin/env python
#
#  Copyright jnw@hachisec.com 2025-09-30
#


import argparse
import json
import random
from bsearch import bsearch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def read_predictions(filenames, truth_dict):
    # truth_dict:
    #  Dictionary mapping predictions idx values to 1 (target) or 0 (benign)
    #
    # file:
    # A file of predictions from a model on its test set
    # File or tab-separated triples (idx, decision, decision_statistic
    #  where idx is the object index from the truth file
    #  decision is the label at threshold 0.5 (0 or 1)
    #  decision_statistic is the model output value in range [0,1]
    #
    # returns idxs, decisions, num_targets, num_nontargets
    # idxs is the list of target indexes
    # decisions is the list of decision statistics corresponding to the idxs in order
    num_targets = 0
    num_nontargets = 0
    decisions = []
    idxs = []
    for filename in filenames:
        try:
            fd = open(filename,'r')
        except:
            print(f'Unable to open predictions file {filename}')
        data = fd.readlines()
        fd.close()
            
        for datum in data:
            tuple = datum.split('\t')
            idx = int(tuple[0])
            #assert not idx in decisions, \
            #    f'duplicate ids in {file}: {idx}'
            if truth_dict[idx]:
                num_targets += 1
            else:
                num_nontargets += 1
            idxs.append(idx)
            decisions.append(float(tuple[2]))
                    
    return idxs, decisions, num_targets, num_nontargets

def grab_truth(truth_files):
    truth = {}
    for file in truth_files:
        fp = open(file)
        num_lines = 0
        for line in fp:
            num_lines += 1
            instance = json.loads(line)
            truth[instance["idx"]] = instance["target"]
        assert num_lines == len(truth.keys()), 'Oops! duplicate idxs!'
    return truth

def start_plot(title, xlabel, ylabel):
    this_fig, this_axes = plt.subplots(figsize=(6,6))
    this_axes.set_title(title)
    this_axes.set_xlabel(xlabel)
    this_axes.set_ylabel(ylabel)
    this_axes.grid(visible=True)
    this_axes.set_xticks(np.arange(0, 1.1, 0.1))
    this_axes.set_yticks(np.arange(0, 1.1, 0.1))
    return this_fig, this_axes

def main():
    parser = argparse.ArgumentParser(usage='Find the harmonic mean or the results of two algorithms on the same dat a')
    parser.add_argument('--truth',
                        nargs='+',
                        help = 'truth_files',
                        required=True)
    parser.add_argument('--p1',
                        nargs='+',
                        help='[predictions1.txt]',
                        required=True)
    parser.add_argument('--p2',
                        nargs='+',
                        help='[predictions2.txt]',
                        required=True)

    args = parser.parse_args()
    
    truth_dict=grab_truth(args.truth)
    idx1,p1, num_targets, num_nontargets = \
        read_predictions(args.p1, truth_dict)
    idx2,p2, num_targets2, num_nontargets2 = \
        read_predictions(args.p2, truth_dict);

    roc_fig, roc_axes = start_plot('', "FPR","TPR")
    roc_axes.set_title("ROC Curves - Identical Thresholds Shown")

    threshold = 0.5
    
    fpr1, tpr1, thresh1 = roc_curve(list(map(lambda x: truth_dict[x], idx1)),
                                    p1)
    thresh1_index = bsearch(thresh1, threshold, op=(lambda a,b: a >= b))
    roc_axes.scatter(fpr1[thresh1_index], tpr1[thresh1_index], marker='x', label='_nolegend_')
    roc_axes.plot(fpr1, tpr1)

    p2 = list(map(lambda x: 1-(1-x*0.48)+0.48, p2))
    fpr2, tpr2, thresh2 = roc_curve(list(map(lambda x: truth_dict[x], idx2)),
                                    p2)
    thresh2_index = bsearch(thresh2, threshold, op=(lambda a,b: a >= b))
    roc_axes.scatter(fpr2[thresh2_index], tpr2[thresh2_index], marker='x')
    roc_axes.plot(fpr2, tpr2)
    
    plt.show(block=True)                          
    

    
if __name__ == "__main__":
    main()
