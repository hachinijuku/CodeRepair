#! /usr/bin/env python
#
#  Copyright jnw@hachisec.com 2025-09-30
#

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import json
from bsearch import bsearch

def monotone_nondecreasing_hull(f):
    for index in range(0, len(f)-1):
        if f[index+1] < f[index]:
            f[index+1] = f[index]
    return f

def find_fpr_tpr(target_precision, fpr, tpr, threshes, num_targets, num_nontargets):
    precisions = []
    
    # For each threshold, find the precision that a collection of samples
    # with that threshold (or greater) has in the validation set,
    # that is, I find the
    # True Positives / (True Positives + False Positives) in that set.
    #
    # Then sort the thresholds and the validation set precision values
    # so they can be use to interpolate test set threshold
    # precision values
    for fprate,tprate,thresh in zip(fpr[1:],
                                    tpr[1:],
                                    threshes[1:]):
        precisions.append(tprate*num_targets/ \
                          (tprate*num_targets+fprate*num_nontargets))
    rev_thresh = threshes[1:].tolist()
    rev_thresh.reverse()
    fpr = fpr[1:]
    precisions.reverse()
    precisions = monotone_nondecreasing_hull(precisions)

    index = bsearch(precisions, 0.2, op=(lambda a, b: a<=b))
    graph_idx = bsearch(threshes[1:], rev_thresh[index], op=(lambda a,b: a >=b))
    return fpr[graph_idx], tpr[graph_idx], threshes[graph_idx]

def start_plot(xlabel, ylabel):
    this_fig, this_axes = plt.subplots()
    this_axes.set_xlabel(xlabel)
    this_axes.set_ylabel(ylabel)
    this_axes.grid(visible=True)
    this_axes.set_xticks(np.arange(0, 1.1, 0.1))
    this_axes.set_yticks(np.arange(0, 1.1, 0.1))
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

def write_predictions(filename, test_idxs, decisions):
    try:
        fd = open(filename,'w')
        for (idx, decision) in \
                zip(test_idxs, decisions):
            print(f'{idx}\t0\t{decision}',file=fd)
        print(f'Writing cal predictions file: {filename}')
    except:
        print(f'Unable to write cal predictions file {filename}')
        fd.close()



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
    parser.add_argument('--l1',
                        default='Algorithm 1',
                        required=False,
                        help='Label for Predictions 1')
    parser.add_argument('--p2',
                        nargs='+',
                        help='[predictions2.txt]',
                        required=True)
    parser.add_argument('--l2',
                        default='Algorithm 2',
                        required=False,
                        help='Label for Predictions 2')
    parser.add_argument('-w',
                        action='store_true',
                        help='write fusion files')
    
    args = parser.parse_args()
    
    truth_dict=grab_truth(args.truth)
    idx1,p1, num_targets, num_nontargets = \
        read_predictions(args.p1, truth_dict)
    idx2,p2, num_targets2, num_nontargets2 = \
        read_predictions(args.p2, truth_dict);
    assert (num_targets == num_targets2) and \
        (num_nontargets == num_nontargets2), \
        '** Number of targets or nontargets mismatch: \n' \
        + f'   {num_targets}:{num_targets2}, {num_nontargets}:{num_nontargets2}'
    
    p1dict = dict(zip(idx1,p1))
    p2dict = dict(zip(idx2,p2))

    idxtrue = list(filter (lambda x: truth_dict[x], idx1))
    idxfalse = list(filter (lambda x: not truth_dict[x], idx1))
    
    scatter_fig, scatter_axes = start_plot(args.l1, args.l2)
    scatter_axes.set_title(f'{args.l2} vs \n {args.l1}')
    
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
    auc1=roc_auc_score(list(map(lambda x: truth_dict[x], idx1)), p1)

    
    roc_axes.plot(fpr1, tpr1)
    fpr_20, tpr_20, thresh20= find_fpr_tpr(0.2, fpr1, tpr1, thresh1, num_targets, num_nontargets)
    roc_axes.scatter(fpr_20, tpr_20, marker='x', label='_nolegend_')
    print(f'{args.p1} = 20% Precision at TPR {tpr_20}')
    
    fpr2, tpr2, thresh2 = roc_curve(list(map(lambda x: truth_dict[x], idx2)),
                                    p2)
    auc2=roc_auc_score(list(map(lambda x: truth_dict[x], idx2)), p2)
    fpr_20, tpr_20, thresh20= find_fpr_tpr(0.2, fpr2, tpr2, thresh2, num_targets, num_nontargets)
    roc_axes.scatter(fpr_20, tpr_20, marker='x', label='_nolegend_')
    print(f'{args.p2} = 20% Precision at TPR {tpr_20}')

    roc_axes.plot(fpr2, tpr2)
    
    p2_in_p1_order = list(map(lambda x: p2dict[x], idx1))
    hmeans = list(map(lambda x, y: 2/(1/x + 1/y), p1, p2_in_p1_order))
    hm_fpr, hm_tpr, hm_thresh = \
        roc_curve(list(map(lambda x: truth_dict[x], idx1)),
                  hmeans)
    hm_auc=roc_auc_score(list(map(lambda x: truth_dict[x], idx1)), hmeans)

    roc_axes.plot(hm_fpr, hm_tpr)
    fpr_20, tpr_20, thresh20= find_fpr_tpr(0.2, hm_fpr, hm_tpr, hm_thresh, num_targets, num_nontargets)
    roc_axes.scatter(fpr_20, tpr_20, marker='x', label='_nolegend_')
    print(f'Harmonic Mean = 20% Precision at TPR {tpr_20}')
    if args.w:
        write_predictions('hmean-predictions.txt', idx1, hmeans)

    maxp1 = max(p1)
    maxp2 = max(p2)
    maxp = list(map(lambda x1, x2: max([x1/maxp1,x2/maxp2]), p1, p2_in_p1_order))
    max_fpr, max_tpr, max_thresh = \
        roc_curve(list(map(lambda x: truth_dict[x], idx1)),
                  maxp)
    max_auc=roc_auc_score(list(map(lambda x: truth_dict[x], idx1)), maxp)
    roc_axes.plot(max_fpr, max_tpr)
    fpr_20, tpr_20, thresh20= find_fpr_tpr(0.2, max_fpr, max_tpr, max_thresh, num_targets, num_nontargets)
    roc_axes.scatter(fpr_20, tpr_20, marker='x', label='_nolegend_')
    print(f'Max = 20% Precision at TPR {tpr_20}')
    if args.w:
        write_predictions('max-predictions.txt', idx1, maxp)

    meanp = list(map(lambda x1, x2: (x1 + x2)/2, p1, p2_in_p1_order))
    mean_fpr, mean_tpr, mean_thresh = \
        roc_curve(list(map(lambda x: truth_dict[x], idx1)), meanp)
    mean_auc=roc_auc_score(list(map(lambda x: truth_dict[x], idx1)), meanp)
    roc_axes.plot(mean_fpr, mean_tpr)
    fpr_20, tpr_20, thresh20= find_fpr_tpr(0.2, mean_fpr, mean_tpr, mean_thresh, num_targets, num_nontargets)
    roc_axes.scatter(fpr_20, tpr_20, marker='x', label='_nolegend_')
    print(f'Mean = 20% Precision at TPR {tpr_20}')
    if args.w:
        write_predictions('mean-predictions.txt', idx1, meanp)

    roc_axes.plot([0,1],[0,1], color='black', linestyle='dashed')
    roc_axes.legend([f'{args.l1} (AUC {auc1:.3f})', f'{args.l2} (AUC {auc2:.3f})', f'Harmonic Mean (AUC {hm_auc:.3f})', f'Max (AUC {max_auc:.3f})', f'Arithmetic Mean (AUC {mean_auc:.3f})', 'random'], loc='lower right')

    
    roc_fig.show()
    
    plt.show(block=True)                          
    
    
if __name__ == "__main__":
    main()
