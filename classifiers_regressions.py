#####################################################################################################################
# classifiers_regressions: This module contains basic metrics for classification and regression                     #
#                          It is part of the Cuicuilco framework                                                    #
# Functions: mean_average_error, classification_rate
#                                                                                                                   #
# By Alberto Escalante. Alberto.Escalante@ini.rub.de                                                                #
# Ruhr-University-Bochum, Institute for Neural Computation, Group of Prof. Dr. Wiskott                              #
#####################################################################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy
import mdp
import patch_mdp


def mean_average_error(ground_truth, regression, verbose=False):
    """ Computes the mean average error (MAE).
     
    Args:    
        ground_truth (list or 1-dim ndarray): ground truth labels.
        regression (list or 1-dim ndarray): label estimations. Must have the same length as the ground truth.
        verbose (bool): verbosity parameter.
    Returns:
        (float): the MAE.
    """
    if len(ground_truth) != len(regression):
        ex = "ERROR in regression labels in mean_average_error:" + \
             "len(ground_truth)=%d != len(regression)=%d" % (len(ground_truth), len(regression))
        print(ex)
        raise Exception(ex)

    d1 = numpy.array(ground_truth).flatten()
    d2 = numpy.array(regression).flatten()
    if verbose:
        print("ground_truth=", d1)
        print("regression=", d2)
    mae = numpy.abs(d2 - d1).mean()
    return mae


def classification_rate(ground_truth, classified, verbose=False):
    """ Computes the classification rate (i.e., fraction of samples correctly classified) 

    Args:    
        ground_truth (list or 1-dim ndarray): ground truth classes. Assumed to be integer.
        classified (list or 1-dim ndarray): class estimations. Assumed to be integer. Must have the 
                                            same length as the ground truth.
        verbose (bool): verbosity parameter.
    Returns:
        (float): the classification rate (i.e., number of successful classifications / total number of samples)
    """
    num = len(ground_truth)
    if len(ground_truth) != len(classified):
        ex = "ERROR in class sizes, in classification_rate:" + \
             "len(ground_truth)=%d != len(classified)=%d" % (len(ground_truth), len(classified))
        raise Exception(ex)

    d1 = numpy.array(ground_truth, dtype="int")
    d2 = numpy.array(classified, dtype="int")
    if verbose:
        print("ground_truth=", d1)
        print("classified=", d2)
    return (d1 == d2).sum() * 1.0 / num
