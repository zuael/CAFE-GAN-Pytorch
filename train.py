import argparse
import datetime
import json
import os
from os.path import join

import torch.utils.data as data

import torch
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from models.cafegan import CAFEGAN
from data import check_attribute_conflict
from helpers import Progressbar, add_scalar_dict
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator


attrs_default = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
]

def parse(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA', 'CelebA-HQ'], default='CelebA')
    parser.add_argument('--data_path', dest='data_path', type=str, default="")
    parser.add_argument('--attr_path', dest='attr_path', type=str, default="")
    parser.add_argument('--image_list_path', dest='image_list_path', type=str, default="")

    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--use_stu', dest='use_stu', action='store_true')
    parser.add_argument('--kernel_size', dest='kernel_size', type=int, default=3)
    parser.add_argument('--stu_norm_fn', dest='stu_norm_fn', type=str, default="none")

    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=3)
    parser.add_argument('--one_more_conv', dest='one_more_conv', action='store_true')
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--fe_layers', dest='fe_layers', type=int, default=3)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--lambda_Datt', dest='lambda_Datt', type=float, default=1.0)
    parser.add_argument('--lambda_Dcls', dest='lambda_Dcls', type=float, default=1.0)
    parser.add_argument('--lambda_Gcls', dest='lambda_Gcls', type=float, default=10.0)
    parser.add_argument('--lambda_rec', dest='lambda_rec', type=float, default=100.0)
    parser.add_argument('--lambda_CM', dest='lambda_CM', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)

    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epochs')
    parser.add_argument('--is_resume', dest='is_resume', type=bool, default=False)
    parser.add_argument('--event_name', dest='event_name', type=str, default=None)
    parser.add_argument('--load_epoch', dest='load_epoch', type=int, default=0)

    parser.add_argument('--batch_size', dest='batch_size', type=int, default=4)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')

    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=16, help='# of sample images')

    parser.add_argument('--save_interval', dest='save_interval', type=int, default=1000)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=1000)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--data_save_root', dest='data_save_root', type=str, default='output')
    parser.add_argument('--experiment_name', dest='experiment_name', default=datetime.datetime.now().strftime("%I-%M%p_on_%B_%d,%Y"))

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse()
    print(args)

    args.lr_base = args.lr
    args.n_attrs = len(args.attrs)
    args.betas = (args.beta1, args.beta2)

    os.makedirs(join(args.data_save_root, args.experiment_name), exist_ok=True)
    os.makedirs(join(args.data_save_root, args.experiment_name, 'checkpoint'), exist_ok=True)
    os.makedirs(join(args.data_save_root, args.experiment_name, 'sample_training'), exist_ok=True)
    with open(join(args.data_save_root, args.experiment_name, 'setting.txt'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

    if args.data == 'CelebA':
        from data import CelebA
        train_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'train', args.attrs)
        valid_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'valid', args.attrs)
    if args.data == 'CelebA-HQ':
        from data import CelebA_HQ
        train_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'train', args.attrs)
        valid_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'valid', args.attrs)
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, drop_last=True
    )
    valid_dataloader = data.DataLoader(
        valid_dataset, batch_size=args.n_samples, num_workers=args.num_workers,
        shuffle=False, drop_last=False
    )
    print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

    cafegan = CAFEGAN(args)

    if args.is_resume:
        cafegan.load(os.path.join(args.data_save_root, args.experiment_name, 'checkpoint','weights.'+str(args.load_epoch)+'.pth'))
        ea = event_accumulator.EventAccumulator(join(args.data_save_root, args.experiment_name, 'summary', args.event_name))
        ea.Reload()
        it = ea.scalars.Items('D/d_loss')[-1][1]
    else:
        it = 0
    cudnn.benchmark = True

    progressbar = Progressbar()

    writer = SummaryWriter(join(args.data_save_root, args.experiment_name, 'summary'))

    fixed_img_real, fixed_label_org = next(iter(valid_dataloader))
    fixed_img_real = fixed_img_real.cuda() if args.gpu else fixed_img_real
    fixed_label_org = fixed_label_org.cuda() if args.gpu else fixed_label_org
    fixed_label_org = fixed_label_org.type(torch.float)
    sample_attr_list = [fixed_label_org]
    for i in range(args.n_attrs):
        tmp = fixed_label_org.clone()
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        sample_attr_list.append(tmp)

    it_per_epoch = len(train_dataset) // args.batch_size
    for epoch in range(args.load_epoch, args.epochs):
        # train with base lr in the first 100 epochs
        # and half the lr in the last 100 epochs
        lr = args.lr_base / 1.0 if epoch <100 else args.lr_base / 2.0
        cafegan.set_lr(lr)

        writer.add_scalar('LR/learning_rate', lr, it+1)
        for img_real, label_org in progressbar(train_dataloader):
            cafegan.train()

            img_real = img_real.cuda() if args.gpu else img_real
            label_org = label_org.cuda() if args.gpu else label_org

            # generate target domain labels randomly
            idx = torch.randperm(len(label_org))
            label_trg = label_org[idx].contiguous()

            c_org = label_org.clone()
            c_trg = label_trg.clone()

            c_org = c_org.cuda() if args.gpu else c_org
            c_trg = c_trg.cuda() if args.gpu else c_trg
            attr_diff = (c_trg - c_org) * args.thres_int

            if (it+1) % (args.n_d+1) != 0:
                errD = cafegan.trainD(img_real, label_org, attr_diff)
                add_scalar_dict(writer, errD, it+1, 'D')
            else:
                errG = cafegan.trainG(img_real, label_trg, attr_diff)
                add_scalar_dict(writer, errG, it+1, 'G')
                if 'errD' in vars():
                    progressbar.say(epoch=epoch, iter=it+1, d_loss=errD['d_loss'], g_loss=errG['g_loss'])
                else:
                    pass

            if (it+1) % args.save_interval == 0:
                # To save storage space, I only checkpoint the weights of G.
                # If you'd like to keep weights of G, D, optim_G, optim_D,
                # please use save() instead of saveG().
                cafegan.save(os.path.join(
                    args.data_save_root, args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
                ))

            if (it+1) % args.sample_interval == 0:
                cafegan.eval()
                with torch.no_grad():
                    samples = [fixed_img_real]
                    for i, att_trg in enumerate(sample_attr_list):
                        attr_diff = (att_trg - sample_attr_list[0]) * args.thres_int
                        samples.append(cafegan.G(fixed_img_real, attr_diff))
                    samples = torch.cat(samples, dim=3)
                    writer.add_image('sample', vutils.make_grid(samples, nrow=1, normalize=True, range=(-1., 1.)), it+1)
                    vutils.save_image(samples, os.path.join(
                            args.data_save_root, args.experiment_name, 'sample_training',
                            'Epoch_({:d})_({:d}of{:d}).jpg'.format(epoch, it%it_per_epoch+1, it_per_epoch)
                        ), nrow=1, normalize=True, range=(-1., 1.))
            it += 1