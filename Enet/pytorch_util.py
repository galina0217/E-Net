from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)

def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)

def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)


class MySpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)

        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))

        if ctx.needs_input_grad[0]:
            grad_matrix1 = Variable(torch.mm(grad_output.data, dense_mat.data.t()).to_sparse())

        return grad_matrix1, grad_matrix2

def gnn_spmm(sp_mat, dense_mat):
    result = MySpMM.apply(sp_mat, dense_mat)
    result.retain_grad()
    return result

class MySpMMNOGRAD(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)

        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        assert not ctx.needs_input_grad[1]
        # if ctx.needs_input_grad[1]:
        #     grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))
        if ctx.needs_input_grad[0]:
            grad_matrix1 = Variable(torch.mm(grad_output.data, dense_mat.data.t()).to_sparse())

        return grad_matrix1, grad_matrix2

def gnn_spmm_nograd(sp_mat, dense_mat):
    result = MySpMMNOGRAD.apply(sp_mat, dense_mat)
    result.retain_grad()
    return result

class MySpMUL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat1, sp_mat2):
        ctx.save_for_backward(sp_mat1, sp_mat2)
        return sp_mat1 * sp_mat2

    @staticmethod
    def backward(ctx, grad_output):
        sp_mat1, sp_mat2 = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(sp_mat1.data * grad_output.data)

        if ctx.needs_input_grad[0]:
            grad_matrix1 = Variable(grad_output.data * sp_mat2.data)

        return grad_matrix1, grad_matrix2

def gnn_spmul(sp_mat1, sp_mat2):
    result = MySpMUL.apply(sp_mat1, sp_mat2)
    result.retain_grad()
    return result


class MySpADD(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat1, sp_mat2):
        ctx.save_for_backward(sp_mat1, sp_mat2)
        return sp_mat1 + sp_mat2

    @staticmethod
    def backward(ctx, grad_output):
        sp_mat1, sp_mat2 = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(grad_output.data)

        if ctx.needs_input_grad[0]:
            grad_matrix1 = Variable(grad_output.data)

        return grad_matrix1, grad_matrix2

def gnn_spadd(sp_mat1, sp_mat2):
    result = MySpADD.apply(sp_mat1, sp_mat2)
    result.retain_grad()
    return result



class MySpMINUS(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat1, sp_mat2):
        ctx.save_for_backward(sp_mat1, sp_mat2)
        return sp_mat1 - sp_mat2

    @staticmethod
    def backward(ctx, grad_output):
        sp_mat1, sp_mat2 = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(-1*grad_output.data)

        if ctx.needs_input_grad[0]:
            grad_matrix1 = Variable(grad_output.data)

        return grad_matrix1, grad_matrix2

def gnn_spminus(sp_mat1, sp_mat2):
    result = MySpMINUS.apply(sp_mat1, sp_mat2)
    result.retain_grad()
    return result


class MySpMAT(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat1, scalar, index):
        ctx.save_for_backward(sp_mat1, scalar, index)
        return sp_mat1 * scalar[index].squeeze()

    @staticmethod
    def backward(ctx, grad_output):
        sp_mat1, scalar, index = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = grad_matrix3 = None

        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            # grad_matrix2 = Variable(torch.empty(scalar.shape))
            grad_matrix2 = Variable(torch.zeros(scalar.shape))
            grad_matrix2[index] = torch.sum((sp_mat1.data * grad_output.data).to_dense())
            grad_matrix2 = grad_matrix2.cuda()

        return grad_matrix1, grad_matrix2, grad_matrix3

def gnn_spmat(sp_mat1, scalar, index):
    index = Variable(torch.LongTensor([index]))
    result = MySpMAT.apply(sp_mat1, scalar, index)
    result.retain_grad()
    return result


class MySpADDSCALAR(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat1, scalar):
        ctx.save_for_backward(sp_mat1, scalar)
        res = torch.sparse.FloatTensor(sp_mat1._indices(),
                                       sp_mat1._values() + scalar.squeeze(),
                                       sp_mat1.shape)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        sp_mat1, scalar = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[1]:
            grad_matrix2 = torch.sparse.sum(grad_output.data).unsqueeze(-1).unsqueeze(-1)

        if ctx.needs_input_grad[0]:
            grad_matrix1 = Variable(grad_output.data)

        return grad_matrix1, grad_matrix2

def gnn_spaddscalar(sp_mat1, scalar):
    result = MySpADDSCALAR.apply(sp_mat1, scalar)
    result.retain_grad()
    return result


class MySpSIGMOID(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat):
        ctx.save_for_backward(sp_mat)
        res = torch.sparse.FloatTensor(sp_mat._indices(),
                                       torch.sigmoid(sp_mat._values()),
                                       sp_mat.shape)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        sp_mat = ctx.saved_variables[0]
        res = torch.sparse.FloatTensor(sp_mat._indices(),
                                       torch.sigmoid(sp_mat._values()),
                                       sp_mat.shape)
        grad = (res._values() * (1 - res._values()))
        grad_matrix = torch.sparse.FloatTensor(sp_mat._indices(),
                                               grad, sp_mat.shape)
        grad_matrix = grad_matrix * grad_output.data
        return grad_matrix

def gnn_spsigmoid(sp_mat):
    result = MySpSIGMOID.apply(sp_mat)
    result.retain_grad()
    return result