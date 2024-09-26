#!/usr/bin/env python


##
# @dir
# In this directory, classes of the controller module are defined and implemented.


##
# In this file, Python functionality for automatic rating of Plexe result files on the mean deviation
# from the pre-defined gap is defined.
# To achieve this, the OMNeT++ Python API omnetpp.scave is used.
# Multithreading is introduced to speed up the processing of multiple evaluations.
# Wrapped by ConstantHeadway class.
#
# @file constant_headway.py
# @ingroup constant_headway
# @author Per Natzschka

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.append(f'{os.environ["OMNETPP_HOME"]}/python')

# noinspection PyUnresolvedReferences
from omnetpp.scave import results as res, vectorops as ops, utils


def get_constant_headway(run_ids: list) -> np.float128:
    ##
    # Calculates a value rating the mean deviation of all vehicles from the pre-defined gap.
    # It calculates the mean squared deviation of each vehicle from its pre-defined gap,
    # adds that value up for each vehicle of a particular run and calculates the mean over all runs
    # (i.e., all repetitions and scenarios).
    #
    # @param run_ids: List of strings representing the OMNeT++ run ids of all runs to be evaluated.
    # @return A longfloat rating the deviation from the pre-defined gap.

    # iterate over all simulation runs
    values = np.empty(len(run_ids), dtype=np.float128)
    for i in range(len(run_ids)):
        # get gap vectors of all but the first vehicles simulated in the given run
        name_filter = f"run =~ {run_ids[i]} AND controlError AND module =~ **.node[*].appl AND NOT module =~ **.node[0]" \
                      f".appl"
        # fetch vectors
        vecs = res.get_vectors(name_filter)     
        # calculate square error on each value
        vecs = ops.expression(vecs, f"y ** 2")
        # calculate means over vectors
        means = np.array([v[1]["vecvalue"].mean(dtype=np.float128) for v in vecs.iterrows()])
        # calculate the sum of means and add to values
        values.put(i, means.sum(dtype=np.float128))
    # return mean over all simulation runs
    return values.mean(dtype=np.float128)


def multithreaded(threads: int, directory: str, run_ids: list) -> list:
    ##
    # Runs get_constant_headway concurrently for multiple simulation
    # results with no more than the given number of threads.
    # This is the function actually called by ConstantHeadway.
    #
    # @param threads: Maximum number of threads to be used for concurrent execution.
    # @param directory: A path to the
    # directory directly or indirectly containing all result files that are to be evaluated.
    # @param run_ids: A list of lists of strings where each list of strings contains all OMNeT++ run ids of the runs
    # conducted for one parameterCombination
    # @return A list of longfloats representing the rating of the given simulation runs.

    res.set_inputs(directory)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = pool.map(get_constant_headway, run_ids)
    return list(results)
