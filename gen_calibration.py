#! /usr/bin/env python
#
#  Copyright jnw@hachisec.com 2025-09-30
#
# gen_calibrations.py
#
# usage: gen_calibrations.py --truth [truth_files] --predictions [test_set_predictions_files] --validations [validation_set_predictions_files]
# Check the code -- you'll get it
#
# example:  gen_calibration.py --truth truth.jsonl --predictions predictions*.txt --validations valid*.txt

import argparse
import json
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

class CalibratedResults:

    def __init__(self, truth_dict, test_decisions_file, validation_decisions_file):
        ##
        # truth_dict:
        #  Dictionary mapping predictions idx values to 1 (target) or 0 (benign)
        #
        # predictions_file:
        # A file of predictions from a model on its test set
        # File or tab-separated triples (idx, decision, decision_statistic
        #  where idx is the object index from the truth file
        #  decision is the label at threshold 0.5 (0 or 1)
        #  decision_statistic is the model output value in range [0,1]
        #
        # validations_file:
        # A file of predictions from a model on its validation set
        
        self.test_decisions_file = test_decisions_file
        self.validation_decisions_file = validation_decisions_file

        # Read both test and validation set decisions
        self.test_idxs, self.test_decisions, self.test_num_targets = \
            self.__read_decisions(truth_dict, self.test_decisions_file)
        self.validation_idxs, self.validation_decisions, self.validation_num_targets = \
            self.__read_decisions(truth_dict, self.validation_decisions_file)

            
        # Get ROCs for both test set decisions and validation set decisions
        self.test_fpr, self.test_tpr, self.test_thresholds = \
            roc_curve(list(map(lambda x:truth_dict[x],
                               list(self.test_idxs))),
                      list(self.test_decisions))

        
        self.validation_fpr, self.validation_tpr, self.validation_thresholds = \
            roc_curve(list(map(lambda x:truth_dict[x], 
                               list(self.validation_idxs))),
                      list(self.validation_decisions))

        precisions = []
        num_nontargets = len(self.validation_decisions) - self.validation_num_targets

        # For each threshold, find the precision that a collection of samples
        # with that threshold (or greater) has in the validation set,
        # that is, I find the
        # True Positives / (True Positives + False Positives) in that set.
        #
        # Then sort the thresholds and the validation set precision values
        # so they can be use to interpolate test set threshold
        # precision values
        for fprate,tprate,thresh in zip(self.validation_fpr[1:],
                                        self.validation_tpr[1:],
                                        self.validation_thresholds[1:]):
            precisions.append(tprate*self.validation_num_targets/ \
                              (tprate*self.validation_num_targets+fprate*num_nontargets))
        rev_thresh = self.validation_thresholds[1:].tolist()
        rev_thresh.reverse()
        self.validation_precisions = precisions[:]
        self.validation_precisions_fpr = self.validation_fpr[1:]
        precisions.reverse()
        precisions = monotone_nondecreasing_hull(precisions)
        
        # Find calibrated test_decisions by mapping the test decision
        # values to their corresponding validation set nondecreasing precisions
        self.test_precision_decisions = np.interp(list(self.test_decisions),
                                             rev_thresh,
                                             precisions).tolist()
        self.calibrated_fpr, self.calibrated_tpr, self.calibrated_thresholds =\
            roc_curve(list(map(lambda x:truth_dict[x],
                               list(self.test_idxs))),
                      list(self.test_precision_decisions))
        
    def __read_decisions(self, truth_dict, file):
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
        
        try:
            fd = open(file,'r')
        except:
            print(f'Unable to open predictions file {file}')
            fd.close()
        data = fd.readlines()
        fd.close()

        idxs = []
        decisions = []
        num_targets = 0
        for datum in data:
            tuple = datum.split('\t')
            idx = int(tuple[0])
            #assert not idx in decisions, \
            #    f'duplicate ids in {file}: {idx}'
            num_targets += int(truth_dict[idx] == 1)
            idxs.append(idx)
            decisions.append(float(tuple[2]))

        return idxs, decisions, num_targets

def write_cal_predictions(filename, cal_object, fpr):
    try:
        fd = open(filename,'w')
        for (idx, test_precision_decision) in \
                zip(cal_object.test_idxs, cal_object.test_precision_decisions):
            print(f'{idx}\t{1 if test_precision_decision>=fpr else 0}\t{test_precision_decision}',file=fd)
        print(f'Writing cal predictions file: {filename}')
    except:
        print(f'Unable to write cal predictions file {filename}')
        fd.close()
        
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
    this_fig, this_axes = plt.subplots()
    this_axes.set_title(title)
    this_axes.set_xlabel(xlabel)
    this_axes.set_ylabel(ylabel)
    this_axes.grid(visible=True)
    return this_fig, this_axes

def monotone_nondecreasing_hull(f):
    for index in range(0, len(f)-1):
        if f[index+1] < f[index]:
            f[index+1] = f[index]
    return f
        
def f1(tpr,fpr,num_targets):
    tp = tpr*num_targets
    fp = fpr*(len(tpr)-num_targets)
    fn = (1-tpr)*num_targets
    return list(map(lambda tpx, fpx, fnx: 2.0*tpx/(2.0*tpx + fpx + fnx),
                    tp, fp, fn))

def main():
    parser = argparse.ArgumentParser(usage='Takes a sequence of prediction files and generates a python calibration function for them')
    parser.add_argument('--truth', nargs='+', help='sequence of ground truth files')
    parser.add_argument('--predictions', nargs='+', help='sequence of predictions files to calibrate over', default=[])
    parser.add_argument('--validations', nargs='+', help='sequene of validation prediction files', default=[])
    parser.add_argument('--calibration_fpr', type=float, help='false positive rate to set calibration prediction threshold', default='0.02')
    
    args = parser.parse_args()
    assert len(args.predictions) == len(args.validations), \
        f'Number of predicditions files ({len(args.predictions)}) must match validation files ({len(args.validations)})'

    truth_dict = grab_truth(args.truth)

    ## Create all figures
    legend_text = []
    combined_roc_fig, combined_roc_axes = start_plot('Combined ROCs',
                                                     'FPR',
                                                     'TPR')
    test_roc_fig, test_roc_axes = start_plot('Receiver Operating Characteristic Curves (Test)',
                                             'FPR',
                                             'TPR')
    test_precision_vs_fpr_fig, test_precision_vs_fpr_axes = \
        start_plot('Precision vs FPR Curves',
                   'FPR',
                   'Precision')
    validation_roc_fig, validation_roc_axes = \
        start_plot('Receiver Operating Characteristic Curves (Validation)',
                   'FPR',
                   'TPR')
    multi_f1_threshold_fig, multi_f1_threshold_axes = \
        start_plot('Combined F1 Scores vs. Threshold',
                   'Threshold',
                   'F1 Score')
    threshold_precision_fig, threshold_precision_axes = \
        start_plot('Monotone Nondecreasing Hull of Decision Statistic to Precision Mappings',
                   'Decision Statistic',
                   'Positive Predictive Value vs Threshold')
    
    calibrations = []
    calibrated_decisions = []
    combined_idxs = []
    combined_decisions = []
    combined_num_targets = 0

    # Read test and validation predictions files, ROCs, Calibration results,
    # and a combined calibration result for all the different test set predictions
    for index in range(0, len(args.predictions)):
        calibrations.append(
            CalibratedResults(truth_dict,
                                              args.predictions[index],
                                              args.validations[index]))

        # plot the incremental test and validation set ROCs
        test_roc_axes.plot(calibrations[-1:][0].test_fpr, calibrations[-1:][0].test_tpr)
        test_precision_vs_fpr_axes.plot(calibrations[-1:][0].validation_precisions_fpr, calibrations[-1:][0].validation_precisions)
        legend_text.append(args.predictions[index])
        validation_roc_axes.plot(calibrations[-1:][0].validation_fpr,
                            calibrations[-1:][0].validation_tpr)
        write_cal_predictions('cal-'+args.predictions[index], calibrations[-1:][0],args.calibration_fpr)

        f1_scores = f1(calibrations[-1:][0].test_tpr,
                       calibrations[-1:][0].test_fpr,
                       calibrations[-1:][0].test_num_targets)
        multi_f1_threshold_axes.plot(calibrations[-1:][0].test_thresholds[len(calibrations[-1:][0].test_thresholds)-1:0:-1],
                            f1_scores[len(f1_scores)-1:0:-1])
        threshold_precision_axes.scatter(calibrations[-1:][0].test_decisions[len(calibrations[-1:][0].test_decisions)-1:0:-1],
                                    calibrations[-1:][0].test_precision_decisions[len(calibrations[-1:][0].test_precision_decisions)-1:0:-1],
                                    s=1,
                                    linewidths=1)
        
        combined_idxs += calibrations[-1:][0].test_idxs
        calibrated_decisions += calibrations[-1:][0].test_precision_decisions
        combined_num_targets += calibrations[-1:][0].test_num_targets
        combined_decisions += calibrations[-1:][0].test_decisions
        
    # plot the combined calibrated ROC
    combined_thresh_fpr, combined_thresh_tpr, combined_thesh_thresholds = \
        roc_curve(list(map(lambda x:truth_dict[x],
                           combined_idxs)),
                  combined_decisions)
    combined_fpr, combined_tpr, combined_thresholds = \
        roc_curve(list(map(lambda x:truth_dict[x],
                           combined_idxs)),
                  list(calibrated_decisions))
    combined_roc_axes.plot(combined_thresh_fpr, combined_thresh_tpr)
    combined_roc_axes.plot(combined_fpr, combined_tpr)

    combined_roc_axes.legend(['Threshold','Calibrated'], loc='lower right')
    combined_roc_fig.show()

    
    # plot the test ROCs
    test_roc_axes.legend(legend_text, loc='lower right')
    test_roc_fig.show()

    test_precision_vs_fpr_axes.legend(legend_text, loc='upper right')
    test_precision_vs_fpr_fig.show()

    # plot the validation ROCs
    validation_roc_axes.legend(legend_text, loc='lower right')
    validation_roc_fig.show()

    #plot threshold to F1 for the combined set
    f1_fpr_fig, f1_fpr_axes = start_plot('Combined F1 Scores vs. FPR',
                                         'FPR',
                                         'F1 Score')
    combined_f1_scores = f1(combined_tpr,
                            combined_fpr,
                            combined_num_targets)
    
    f1_fpr_axes.plot(combined_fpr, combined_f1_scores)
    f1_fpr_fig.show()

    multi_f1_threshold_axes.legend(legend_text, loc='lower left')
    multi_f1_threshold_fig.show()

    threshold_precision_axes.legend(legend_text, loc='upper left')
    threshold_precision_fig.show()

    plt.show(block=True)

if __name__ == "__main__":
    main()

