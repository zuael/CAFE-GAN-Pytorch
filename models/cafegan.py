import torch
from .nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock, ConvGRUCell, Dis_Att, Dis_cls
import torch.nn as nn
from torchsummary import summary


MAX_DIM = 64 * 16  # 1024
class Generator(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 stu_norm_fn='none', kernel_size=3, n_attrs=13, shortcut_layers=2, 
                 img_size=128, use_stu=True, one_more_conv=True):
        super(Generator, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers  # f_size = 4 for 128x128
        self.n_attrs = n_attrs
        self.use_stu = use_stu
        self.dec_layers = dec_layers
        self.one_more_conv = one_more_conv
        
        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, kernel_size, stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.Encode = nn.ModuleList(layers)

        if use_stu:
            layers = []
            for i in reversed(range(enc_layers-1-self.shortcut_layers, enc_layers-1)):
                layers += [ConvGRUCell(
                    n_attrs=n_attrs, 
                    state_dim=min(enc_dim*2**(i+1), MAX_DIM),
                    in_dim=min(enc_dim*2**i, MAX_DIM),
                    out_dim=min(dec_dim*2**i, MAX_DIM),
                    norm_fn=stu_norm_fn, 
                    kernel_size=kernel_size, 
                )]
            self.stu = nn.ModuleList(layers)

        layers = []
        n_in = n_in + n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2 ** (dec_layers - i - 1), MAX_DIM)
                if i == 0:
                    layers += [ConvTranspose2dBlock(
                        n_in, n_out, kernel_size, stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                    )]
                elif i <= self.shortcut_layers:
                    n_in = n_in + n_in//2
                    layers += [ConvTranspose2dBlock(
                        n_in, n_out, kernel_size, stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                    )]
                else:
                    layers += [ConvTranspose2dBlock(
                        n_in, n_out, kernel_size, stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                    )]
                n_in = n_out
            else:
                if one_more_conv:
                    layers += [ConvTranspose2dBlock(
                        n_in, dec_dim // 4, kernel_size, stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                    )]
                    layers += [ConvTranspose2dBlock(
                        dec_dim // 4, 3, kernel_size, stride=1, padding=1, norm_fn='none', acti_fn='tanh'
                    )]
                else:
                    layers +=[ConvTranspose2dBlock(
                        n_in, 3, kernel_size, stride=2, padding=1, norm_fn='none', acti_fn='tanh'
                    )]

        self.Decode = nn.ModuleList(layers)
    
    def encode(self, x):
        z = x
        zs = []
        for layer in self.Encode:
            z = layer(z)
            zs.append(z)
        return zs
    
    def decode(self, zs, a):

        out = zs[-1]
        n, _, h, w = out.size()
        attr = a.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h, w))
        out = self.Decode[0](torch.cat([out, attr], dim=1))
        stu_state = zs[-1]

        #propagate shortcut layers
        for i in range(1, self.shortcut_layers + 1):
            if self.use_stu:
                stu_out, stu_state =self.stu[i-1](zs[-(i+1)], stu_state, a)
                out = torch.cat([out, stu_out], dim=1)
                out = self.Decode[i](out)
            else:
                out = torch.cat([out, zs[-(i+1)]], dim=1)
                out = self.Decode[i](out)

        for i in range(self.shortcut_layers + 1, self.dec_layers + 1
        if self.one_more_conv else self.dec_layers):
            out = self.Decode[i](out)

        return out
    
    def forward(self, x, a=None, mode='enc-dec'):
        if mode == 'enc-dec':
            assert a is not None, 'No given attribute.'
            return self.decode(self.encode(x), a)
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        raise Exception('Unrecognized mode: ' + mode)

            
class Discriminators(nn.Module):
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 num_classes=13, fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu',
                 image_size=128, n_layers=5, fe_layers=3, kernel_size=3):
        super(Discriminators, self).__init__()
        assert fe_layers <= n_layers, print('fe_layers > n_layers')
        # feature extractor
        layers = []
        n_in = 3
        for i in range(0, fe_layers):
            n_out = min(dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, kernel_size, stride=2, padding=1,
                norm_fn=norm_fn, acti_fn=acti_fn)]
            n_in = n_out

        self.f_extractor = nn.Sequential(*layers)

        # Datt
        self.Datt = Dis_Att(in_channels=min(dim * 2**(fe_layers-1), MAX_DIM), 
                            norm_fn=norm_fn,  acti_fn=acti_fn,
                            kernel_size=kernel_size, num_classes=num_classes)
        # Dcls
        self.Dcls = Dis_cls(num_tasks=num_classes, n_layers=n_layers,
                            fe_layers=fe_layers, dim=dim, fc_dim=MAX_DIM,
                            fc_norm_fn=fc_norm_fn, fc_acti_fn=fc_acti_fn,
                            norm_fn=norm_fn, acti_fn=acti_fn, kernel_size=kernel_size)
        # Dadv
        layers = []
        n_in = min(dim * 2**(fe_layers-1), MAX_DIM)
        for i in range(fe_layers, n_layers):
            n_out = min(dim * 2 ** i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, kernel_size, stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn),
                ]
            n_in = n_out
        self.adv_conv = nn.Sequential(*layers)

        self.adv_fc = nn.Sequential(
            LinearBlock((image_size//(2**n_layers))**2 * fc_dim, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )

    def forward(self, imgs, type=None):
        feature = self.f_extractor(imgs)

        if type == None:
            mid = self.adv_conv(feature)
            abn_f, abn_p, abn_att, cabn_f, cabn_p, cabn_att=self.Datt(feature)
            cls1_rx, cls2_rx = self.Dcls(feature, abn_att, cabn_att)
            adv = self.adv_fc(mid.view(mid.shape[0], -1))
            return adv.view(adv.shape[0], 1, 1, 1), abn_f, cabn_f, abn_p, cabn_p, cls1_rx, cls2_rx

        elif type == 'adv':
            h = self.adv_conv(feature)
            adv = self.adv_fc(h.view(h.shape[0], -1))
            return adv.unsqueeze(2).unsqueeze(3)

        elif type == 'abn':
            abn_f, _, _, cabn_f, _, _=self.Datt(feature)
            return abn_f, cabn_f

        elif type == 'att':
            _, _, abn_att, _, _, cabn_att=self.Datt(feature)
            return abn_att, cabn_att

import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

class CAFEGAN():
    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.lambda_Datt = args.lambda_Datt
        self.lambda_Dcls = args.lambda_Dcls
        self.lambda_Gcls = args.lambda_Gcls
        self.lambda_rec = args.lambda_rec
        self.lambda_CM = args.lambda_CM
        self.lambda_gp = args.lambda_gp
        
        self.G = Generator(
            enc_dim=args.enc_dim, enc_layers=args.enc_layers, enc_norm_fn=args.enc_norm,
            enc_acti_fn=args.enc_acti, dec_dim=args.dec_dim, dec_layers=args.dec_layers,
            dec_norm_fn=args.dec_norm, dec_acti_fn=args.dec_acti, n_attrs=args.n_attrs,
            shortcut_layers=args.shortcut_layers, img_size=args.img_size, use_stu=args.use_stu,
            kernel_size=args.kernel_size, stu_norm_fn=args.stu_norm_fn, one_more_conv=args.one_more_conv
        )

        self.D = Discriminators(
            dim=args.dis_dim, norm_fn=args.dis_norm, acti_fn=args.dis_acti, kernel_size=args.kernel_size,
            image_size=args.img_size, fc_dim=args.dis_fc_dim, fc_norm_fn=args.dis_fc_norm, num_classes=args.n_attrs,
            fc_acti_fn=args.dis_fc_acti, n_layers=args.dis_layers, fe_layers=args.fe_layers
        )

        self.G.train()
        self.D.train()

        if self.gpu:
            self.G.cuda()
            self.D.cuda()

        summary(self.D, (3, args.img_size, args.img_size),
                batch_size=args.batch_size, device='cuda' if args.gpu else 'cpu')
        summary(self.G, [(3, args.img_size, args.img_size), (args.n_attrs, 1, 1)],
                batch_size=args.batch_size, device='cuda' if args.gpu else 'cpu')

        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)
    
    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr
    
    def trainG(self, img_real, label_trg, attr_diff):
        for p in self.D.parameters():
            p.requires_grad = False

        zs = self.G(img_real, mode='enc')
        img_fake = self.G(zs, attr_diff, mode='dec')
        img_recon = self.G(zs, torch.zeros_like(attr_diff), mode='dec')
        with torch.no_grad():
            abn_f_real, cabn_f_real = self.D(img_real, 'abn')
        adv_fake, abn_f_fake, cabn_f_fake, _, _, cls1_rx_fake, cls2_rx_fake = self.D(img_fake)

        num, n_att, fh, fw = abn_f_real.size()

        if self.mode == 'wgan':
            gf_loss = -adv_fake.mean()
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(adv_fake, torch.ones_like(adv_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(adv_fake, torch.ones_like(adv_fake))

        gc_loss = F.binary_cross_entropy_with_logits(cls1_rx_fake.squeeze(3).squeeze(2),
                                                      label_trg) + \
                   F.binary_cross_entropy_with_logits(cls2_rx_fake.squeeze(3).squeeze(2),
                                                      label_trg)

        gr_loss = F.l1_loss(img_recon, img_real)

        gcm_loss = 0
        for i in range(n_att):
            for j in range(num):
                if attr_diff[j, i] == 0:
                    gcm_loss += F.l1_loss(abn_f_fake[j][i], abn_f_real[j][i]) + \
                                F.l1_loss(cabn_f_fake[j][i], cabn_f_real[j][i])
                else:
                    gcm_loss += F.l1_loss(cabn_f_fake[j][i], abn_f_real[j][i]) + \
                                F.l1_loss(abn_f_fake[j][i], cabn_f_real[j][i])

        gcm_loss /= num * fh * fw

        g_loss = gf_loss + self.lambda_Gcls * gc_loss + \
                 self.lambda_rec * gr_loss + self.lambda_CM * gcm_loss
        
        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()
        
        errG = {
            'g_loss': g_loss.item(), 'gf_loss':gf_loss.item(),'gc_loss': gc_loss.item(),
            'gr_loss': gr_loss.item(), 'gcm_loss': gcm_loss.item()
        }
        return errG
    
    def trainD(self, img_real, label_org, attr_diff):
        for p in self.D.parameters():
            p.requires_grad = True

        with torch.no_grad():
            img_fake = self.G(img_real, attr_diff)
        adv_real, _, _, abn_p_real, cabn_p_real, cls1_rx_real, cls2_rx_real = self.D(img_real)
        adv_fake = self.D(img_fake, type='adv')
        
        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter
            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x, type='adv')
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.shape[0], -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp
        
        if self.mode == 'wgan':
            wd = adv_real.mean() - adv_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_real, img_fake)
        if self.mode == 'lsgan':  # mean_squared_error
            df_loss = F.mse_loss(adv_real, torch.ones_like(adv_fake)) + \
                      F.mse_loss(adv_fake, torch.zeros_like(adv_fake))
            df_gp = gradient_penalty(self.D, img_real)
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            df_loss = F.binary_cross_entropy_with_logits(adv_real, torch.ones_like(adv_real)) + \
                      F.binary_cross_entropy_with_logits(adv_fake, torch.zeros_like(adv_fake))
            df_gp = gradient_penalty(self.D, img_real)

        datt_loss = F.binary_cross_entropy_with_logits(abn_p_real.squeeze(3).squeeze(2),
                                                       label_org) + \
                     F.binary_cross_entropy_with_logits(cabn_p_real.squeeze(3).squeeze(2),
                                                        1 - label_org)

        dc_loss = F.binary_cross_entropy_with_logits(cls1_rx_real.squeeze(3).squeeze(2),
                                                      label_org) + \
                   F.binary_cross_entropy_with_logits(cls2_rx_real.squeeze(3).squeeze(2),
                                                      label_org)

        d_loss = df_loss + self.lambda_gp * df_gp + self.lambda_Datt * datt_loss + self.lambda_Dcls * dc_loss

        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()
        
        errD = {
            'd_loss': d_loss.item(),
            'df_loss': df_loss.item(), 'df_gp': df_gp.item(),
            'datt_loss': datt_loss.item(), 'dc_loss': dc_loss.item()
        }

        return errD
    
    def train(self):
        self.G.train()
        self.D.train()
    
    def eval(self):
        self.G.eval()
        self.D.eval()
    
    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict()
        }
        torch.save(states, path)


    def saveG_D(self, path, flag=None):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
        }
        if flag is None:
            torch.save(states, path)
        elif flag == 'unzip':
            torch.save(states, f=path, _use_new_zipfile_serialization=False)
    
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])

    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)

    def saveD(self, path):
        states = {
            'D': self.D.state_dict()
        }
        torch.save(states, path)