import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from graph.ntu_rgb_d import Graph
import torch.nn.functional as F
from model.aux_model import Modelys


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, use_mask=True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        self.use_mask = use_mask
        if use_mask:
            self.mask = nn.Parameter(torch.zeros(A.shape), requires_grad=True)
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            if self.use_mask:
                A = self.PA * self.mask
                A = self.L2_norm(A)
            else:
                A = self.PA
                A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, use_mask=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive, use_mask=use_mask)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, num_set=5, use_mask=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = np.stack([np.eye(num_point)] * num_set, axis=0)
        base_channel = 64
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.graph = graph
        self.graph_args = graph_args
        self.in_channels = in_channels
        self.drop_out = drop_out
        self.adaptive = adaptive
        self.y_model = Modelys(num_class=self.num_class, num_point=self.num_point, num_person=self.num_person,
                               graph=self.graph,
                               graph_args=self.graph_args, in_channels=self.in_channels)

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.z_p = torch.empty(num_class, base_channel * 4)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False, adaptive=adaptive, use_mask=use_mask)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, use_mask=use_mask)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, use_mask=use_mask)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, use_mask=use_mask)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive, use_mask=use_mask)
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, use_mask=use_mask)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, use_mask=use_mask)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive, use_mask=use_mask)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, use_mask=use_mask)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, use_mask=use_mask)
        self.fc = nn.Linear(256, 256)
        self.fc2 = nn.Linear(base_channel * 4 + num_class, base_channel * 4)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        self.classify = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.classify.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def aux_information(self, x, y):

        x_t = (x + F.relu(self.fc2(y)))
        return x_t

    def _vae_trick(self, x_t):
        mu = self.fc_miu(x_t)
        log_std = self.fc_log(x_t)
        return self.latent_sample(mu, log_std)

    def forward(self, x):

        ys, zs = self.y_model(x)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = F.relu(self.fc(x))
        x = self.drop_out(x)
        z = self.aux_information(x, ys)
        y_hat = self.classify(z)

        return y_hat, z, ys, zs
