"""Network components."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from switchable_norm import SwitchNorm1d, SwitchNorm2d

def add_normalization_1d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm1d(n_out))
    elif fn == 'instancenorm':
        layers.append(Unsqueeze(-1))
        layers.append(nn.InstanceNorm1d(n_out, affine=True))
        layers.append(Squeeze(-1))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm1d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_normalization_2d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm2d(n_out))
    elif fn == 'instancenorm':
        layers.append(nn.InstanceNorm2d(n_out, affine=True))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm2d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_activation(layers, fn):
    if fn == 'none':
        pass
    elif fn == 'relu':
        layers.append(nn.ReLU())
    elif fn == 'lrelu':
        layers.append(nn.LeakyReLU())
    elif fn == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif fn == 'tanh':
        layers.append(nn.Tanh())
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers

class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.squeeze(self.dim)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.unsqueeze(self.dim)


class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, norm_fn='none', acti_fn='none'):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn=='none'))]
        layers = add_normalization_1d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class Conv2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, 
                 norm_fn='none', acti_fn='none', bias=None):
        super(Conv2dBlock, self).__init__()

        layers = []
        if kernel_size % 2 == 0 and stride == 1:
            layers.append(torch.nn.ZeroPad2d((0, 1, 0, 1)))
        layers += [nn.Conv2d(n_in, n_out, kernel_size, stride=stride, padding=padding, 
        bias=(norm_fn=='none') if bias is None else bias)]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, 
                 norm_fn='none', acti_fn='none', bias=None):
        super(ConvTranspose2dBlock, self).__init__()
        layers = []
        if stride > 1:
            layers.append(nn.UpsamplingBilinear2d(scale_factor=stride))
        if kernel_size % 2 == 0:
            layers.append(nn.ZeroPad2d((0, 1, 0, 1)))
        layers += [nn.Conv2d(n_in, n_out, kernel_size, padding=padding, 
                            bias=(norm_fn=='none') if bias is None else bias)]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class make_mtl_block(nn.Module):

    def __init__(self, num_tasks=13, n_layers=5, fe_layers=3,
                 fc_norm_fn='none', fc_acti_fn='lrelu',
                 dim=64, fc_dim=1024, norm_fn='instancenorm',
                 acti_fn='lrelu', kernel_size=3):
        self.num_tasks = num_tasks
        super(make_mtl_block, self).__init__()
        layers = []
        n_in = min(dim * 2**(fe_layers-1), fc_dim)
        for i in range(fe_layers, n_layers):
            n_out = min(dim * 2 ** i, fc_dim)
            layers += [Conv2dBlock(
                n_in, n_out, kernel_size, stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn),
                ]
            n_in = n_out

        self.share_conv = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layer = LinearBlock(fc_dim, fc_dim, fc_norm_fn, fc_acti_fn)

        output = [nn.Linear(fc_dim, 1, bias=True) for _ in range(self.num_tasks)]

        self.output = nn.ModuleList(output)

    def forward(self, x, att_elem):
        pred = []
        bs, _, ys, xs = att_elem.shape
        for i in range(self.num_tasks):
            item_att = att_elem[:, i].view(bs, 1, ys, xs)

            sh = item_att * x
            sh = self.share_conv(sh)
            sh = self.avgpool(sh)
            sh = sh.view(sh.size(0), -1)
            sh = self.output[i](self.fc_layer(sh))
            pred.append(sh)

        return torch.cat(pred, dim=1).unsqueeze(2).unsqueeze(3)

class ABN_Block(nn.Module):
    def __init__(self, in_planes, num_classes=13):
        super(ABN_Block, self).__init__()
        self.att_conv = nn.Conv2d(in_planes, num_classes, kernel_size=1, padding=0,
                                  stride=1, bias=False)
        self.att_gap = nn.AdaptiveMaxPool2d((1,1))
        self.sigmoid = nn.Sigmoid()
        self.depth_conv = nn.Conv2d(
            in_channels=num_classes,out_channels=num_classes,
            kernel_size=1,stride=1,padding=0, groups=num_classes)

    def forward(self, ax):

        a_feature = self.att_conv(ax)
        temp = self.depth_conv(a_feature)
        a_probability = self.att_gap(temp)
        att = self.sigmoid(temp)

        return a_feature, a_probability, att

class Dis_Att(nn.Module):

    def __init__(self, in_channels=256, norm_fn='instancenorm',
                 acti_fn='lrelu', num_classes=13, kernel_size=3):
        super(Dis_Att, self).__init__()
        self.share_conv = nn.Sequential(
            Conv2dBlock(in_channels, 512, kernel_size, stride=1, padding=1,
                        norm_fn=norm_fn, acti_fn=acti_fn),
            Conv2dBlock(512, 512, kernel_size, stride=1, padding=1,
                        norm_fn=norm_fn, acti_fn=acti_fn))
        self.abn = ABN_Block(512, num_classes)
        self.cabn = ABN_Block(512, num_classes)

    def forward(self, feature):
        ax = self.share_conv(feature)
        abn_f, abn_p, abn_att = self.abn(ax)
        cabn_f, cabn_p, cabn_att = self.cabn(ax)

        return abn_f, abn_p, abn_att, cabn_f, cabn_p, cabn_att

class Dis_cls(nn.Module):
    def __init__(self, num_tasks=13, n_layers=5, fe_layers=3,
                 fc_norm_fn='instancenorm', fc_acti_fn='lrelu',
                 kernel_size=3, dim=64, fc_dim=1024, 
                 norm_fn='instancenorm', acti_fn='lrelu'):

        super(Dis_cls, self).__init__()
        self.cls1 = make_mtl_block(num_tasks=num_tasks, n_layers=n_layers,
                                   fe_layers=fe_layers, dim=dim, fc_dim=fc_dim,
                                   fc_norm_fn=fc_norm_fn, fc_acti_fn=fc_acti_fn,
                                   norm_fn=norm_fn, acti_fn=acti_fn, kernel_size=kernel_size)
        self.cls2 = make_mtl_block(num_tasks=num_tasks, n_layers=n_layers,
                                   fe_layers=fe_layers, dim=dim, fc_dim=fc_dim,
                                   fc_norm_fn=fc_norm_fn, fc_acti_fn=fc_acti_fn,
                                   norm_fn=norm_fn, acti_fn=acti_fn, kernel_size=kernel_size)

    def forward(self, feature, abn_att, cabn_att):
        cls1_rx = self.cls1(feature, abn_att)
        cls2_rx = self.cls2(feature, cabn_att)

        return cls1_rx, cls2_rx