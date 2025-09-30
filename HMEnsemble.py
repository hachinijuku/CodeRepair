#! /usr/bin/env python
#
#  Copyright jnw@hachisec.com 2025-09-30
#

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import json

def start_plot(xlabel, ylabel):
    this_fig, this_axes = plt.subplots()
    this_axes.set_xlabel(xlabel)
    this_axes.set_ylabel(ylabel)
    this_axes.grid(visible=True)
    return this_fig, this_axes

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
    # returns idxs, decisions, num_targets
    # idxs is the list of target indexes
    # decisions is the list of decision statistics corresponding to the idxs in order
    # num_targets is the number of target objects as given by the truth_dict
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
            idxs.append(idx)
            decisions.append(float(tuple[2]))
            
        
        return idxs, decisions
        

def main():
    parser = argparse.ArgumentParser(usage='Find the harmonic mean or the results of two algorithms on the same dat a')
    parser.add_argument('--truth', nargs='+', help = 'truth_files' )
    parser.add_argument('--p1', nargs='+', help='[predictions1.txt])')
    parser.add_argument('--l1', default='Algorithm 1', required=False,  help='Label for Predictions 1')
    parser.add_argument('--p2', nargs='+', help='[predictions2.txt]')
    parser.add_argument('--l2', default='Algorithm 2', required=False,  help='Label for Predictions 2')
    args = parser.parse_args()
    
    truth_dict=grab_truth(args.truth)
    idx1,p1 = read_predictions(args.p1, truth_dict)
    idx2,p2 = read_predictions(args.p2, truth_dict);

    p1dict = dict(zip(idx1,p1))
    p2dict = dict(zip(idx2,p2))

    idxtrue = list(filter (lambda x: truth_dict[x], idx1))
    idxfalse = list(filter (lambda x: not truth_dict[x], idx1))
    
    scatter_fig, scatter_axes = start_plot(args.l1, args.l2)
    scatter_axes.set_title(f'Scatter Plot {args.l2} vs {args.l1})')
    
    p1false = list(map(lambda x: p1dict[x],idxfalse))
    p2false = list(map(lambda x: p2dict[x], idxfalse))
    scatter_axes.scatter(p1false, p2false, marker='o', color='orange',s=2, linewidth=1)
    p1true = list(map(lambda x: p1dict[x],idxtrue))
    p2true = list(map(lambda x: p2dict[x], idxtrue))
    scatter_axes.scatter(p1true, p2true, marker='*', color='blue',s=2, linewidth=1)

    scatter_axes.legend(['Benign','Vulnerable'],loc='lower right')
    scatter_fig.show()
    
    roc_fig, roc_axes = start_plot("FPR","TPR")
    roc_axes.set_title("ROC Curves")
    
    fpr1, tpr1, thresh1 = roc_curve(list(map(lambda x: truth_dict[x], idx1)),
                                    p1)
    roc_axes.plot(fpr1, tpr1)
    
    fpr2, tpr2, thresh2 = roc_curve(list(map(lambda x: truth_dict[x], idx2)),
                                    p2)
    roc_axes.plot(fpr2, tpr2)
    
    p2_in_p1_order = list(map(lambda x: p2dict[x], idx1))
    hmeans = list(map(lambda x, y: 2/(1/x + 1/y), p1, p2_in_p1_order))
    hm_fpr, hm_tpr, hm_thresh = \
        roc_curve(list(map(lambda x: truth_dict[x], idx1)),
                  hmeans)
    roc_axes.plot(hm_fpr, hm_tpr)

    maxp1 = max(p1)
    maxp2 = max(p2)
    maxp = list(map(lambda x1, x2: max([x1/maxp1,x2/maxp2]), p1, p2_in_p1_order))
    max_fpr, max_tpr, max_thresh = \
        roc_curve(list(map(lambda x: truth_dict[x], idx1)),
                  maxp)
    roc_axes.plot(max_fpr, max_tpr)

    meanp = list(map(lambda x1, x2: (x1 + x2)/2, p1, p2_in_p1_order))
    mean_fpr, mean_tpr, mean_thresh = \
        roc_curve(list(map(lambda x: truth_dict[x], idx1)), meanp)
    roc_axes.plot(mean_fpr, mean_tpr)

    roc_axes.legend([args.l1, args.l2, 'Harmonic Mean', 'Max', 'Arithmetic Mean'], loc='lower right')
    roc_fig.show()
    
    plt.show(block=True)                          
    
    
if __name__ == "__main__":
    main()
