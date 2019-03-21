#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import cython
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libc.math cimport exp, log, pow, sqrt
from libc.string cimport memset

from scipy.linalg.blas import _fblas as fblas

ctypedef np.float64_t REAL_t
ctypedef np.uint32_t  INT_t
ctypedef double (*ddot_ptr) (const int *N, const double *X, const int *incX, const double *Y, const int *incY) nogil
cdef ddot_ptr ddot=<ddot_ptr>PyCObject_AsVoidPtr(fblas.ddot._cpointer)  # vector-vector multiplication


cdef extern from "/home/dina/PycharmProjects/Glove/voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)
cdef int ONE = 1
def inner_product(vec1, vec2, _size):
    cdef int size = _size
    return <REAL_t>ddot(&size, <REAL_t *>(np.PyArray_DATA(vec1)), &ONE, <REAL_t *>(np.PyArray_DATA(vec2)), &ONE)

cdef void train_glove_thread(
        REAL_t * W,       REAL_t * contextW,
        REAL_t * gradsqW, REAL_t * gradsqContextW,
        REAL_t * bias,    REAL_t * contextB,
        REAL_t * gradsqb, REAL_t * gradsqContextB,
        REAL_t * error,
        INT_t * i, INT_t * j, REAL_t * Xij,
        int vectorSize, int batchSize, REAL_t xMax, REAL_t alpha, REAL_t stepSize) nogil:

    cdef long long a, b, l1, l2
    cdef int batchIndex = 0
    cdef REAL_t temp1, temp2, diff, fdiff

    for batchIndex in range(batchSize):
        # Calculate cost, save diff for gradients
        l1 = i[batchIndex]*vectorSize
        l2 = j[batchIndex]*vectorSize

        diff = 0.0;
        for b in range(vectorSize):
            diff += W[b + l1] * contextW[b + l2] # dot product of word and context word vector
        diff += bias[i[batchIndex]] + contextB[j[batchIndex]] - log(Xij[batchIndex])
        fdiff = diff if (Xij[batchIndex] > xMax) else pow(Xij[batchIndex] / xMax, alpha) * diff
        error[0] += 0.5 * fdiff * diff # weighted squared error

        # # Adaptive gradient updates
        fdiff *= stepSize # for ease in calculating gradient
        for b in range(vectorSize):
            # learning rate times gradient for word vectors
            temp1 = fdiff * contextW[b + l2]
            temp2 = fdiff * W[b + l1]
            # adaptive updates
            W[b + l1]              -= (temp1 / sqrt(gradsqW[b + l1]))
            contextW[b + l2]       -= (temp2 / sqrt(gradsqContextW[b + l2]))
            gradsqW[b + l1]        += temp1 * temp1
            gradsqContextW[b + l2] += temp2 * temp2
        # updates for bias terms
        bias[i[batchIndex]]        -= fdiff / sqrt(gradsqb[i[batchIndex]]);
        contextB[j[batchIndex]] -= fdiff / sqrt(gradsqContextB[j[batchIndex]]);

        fdiff *= fdiff;
        gradsqb[i[batchIndex]]           += fdiff
        gradsqContextB[j[batchIndex]] += fdiff

def train_glove(model, jobs, float _stepSize, _error):
    cdef REAL_t *W              = <REAL_t *>(np.PyArray_DATA(model.W))
    cdef REAL_t *contextW       = <REAL_t *>(np.PyArray_DATA(model.contextW))
    cdef REAL_t *gradsqW        = <REAL_t *>(np.PyArray_DATA(model.gradsqW))
    cdef REAL_t *gradsqContextW = <REAL_t *>(np.PyArray_DATA(model.gradsqContextW))

    cdef REAL_t *b              = <REAL_t *>(np.PyArray_DATA(model.b))
    cdef REAL_t *contextB       = <REAL_t *>(np.PyArray_DATA(model.contextB))
    cdef REAL_t *gradsqb        = <REAL_t *>(np.PyArray_DATA(model.gradsqb))
    cdef REAL_t *gradsqContextB = <REAL_t *>(np.PyArray_DATA(model.gradsqContextB))

    cdef REAL_t *error          = <REAL_t *>(np.PyArray_DATA(_error))

    cdef INT_t  *jobCenterWords        = <INT_t  *>(np.PyArray_DATA(jobs[0]))
    cdef INT_t  *jobContextWords     = <INT_t  *>(np.PyArray_DATA(jobs[1]))
    cdef REAL_t *Xij     = <REAL_t *>(np.PyArray_DATA(jobs[2]))

    # configuration and parameters
    cdef REAL_t stepSize = _stepSize
    cdef int vectorSize = model.d
    cdef int batchSize = len(jobs[0])
    cdef REAL_t xMax   = model.xMax
    cdef REAL_t alpha   = model.alpha

    # release GIL & train on the sentence
    with nogil:
        train_glove_thread(
            W,\
            contextW,\
            gradsqW,\
            gradsqContextW,\
            b,\
            contextB,\
            gradsqb,\
            gradsqContextB,\
            error,\
            jobCenterWords,\
            jobContextWords,\
            Xij, \
            vectorSize,\
            batchSize, \
            xMax, \
            alpha, \
            stepSize
        )