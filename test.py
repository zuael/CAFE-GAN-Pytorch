import argparse
import json
import os
from os.path import join
from data import CelebA
import torch
import torch.utils.data as data
import torchvision.utils as vutils
import numpy as np
import cv2
from utils import find_model
from cafegan import CAFEGAN
from data import check_attribute_conflict
import torch.nn.functional as F

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', type=str, default=None, required=True)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--use_model', dest='use_model', type=str, choices=['D', 'G'], default='G')
    parser.add_argument('--mode', dest='mode', type=str, choices=['test', 'fast_test'], default='fast_test')
    parser.add_argument('--custom_img', action='store_true')
    parser.add_argument('--custom_data', type=str, default='./data/custom')
    parser.add_argument('--custom_attr', type=str, default='./data/list_attr_custom.txt')
    parser.add_argument('--data_path', dest='data_path', type=str, default=None)
    parser.add_argument('--attr_path', dest='attr_path', type=str, default=None)
    parser.add_argument('--image_list_path', dest='image_list_path', type=str, default=None)
    parser.add_argument('--data_save_root', dest='data_save_root', type=str, default=None)
    return parser.parse_args(args)


args_ = parse()
with open(join(args_.data_save_root, args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))


if args.custom_img:
    from data import Custom
    test_dataset = Custom(args_.custom_data, args_.custom_attr, args.img_size, args.attrs)
else:
    if args.data == 'CelebA':
        from data import CelebA
        test_dataset = CelebA(args_.data_path, args_.attr_path, args.img_size, args_.mode, args.attrs)
    if args.data == 'CelebA-HQ':
        from data import CelebA_HQ
        test_dataset = CelebA_HQ(args_.data_path, args_.attr_path, args_.image_list_path, args.img_size, args_.mode, args.attrs)

test_dataloader = data.DataLoader(
    test_dataset, batch_size=1, num_workers=1,
    shuffle=False, drop_last=False
)

print('Testing images:', len(test_dataset))

if args_.use_model == 'G':
    output_path = join(args_.data_save_root, args_.experiment_name, 'edit_testing')
    os.makedirs(output_path, exist_ok=True)
elif args_.use_model == 'D':
    output_path = join(args_.data_save_root, args_.experiment_name, 'attention_testing')
    os.makedirs(output_path, exist_ok=True)

cafegan = CAFEGAN(args)
cafegan.load(find_model(join(args_.data_save_root, args_.experiment_name, 'checkpoint'), epoch=args_.load_epoch))
cafegan.eval()

for idx, (img_real, att_org) in enumerate(test_dataloader):
    img_real = img_real.cuda() if args_.gpu else img_real
    att_org = att_org.cuda() if args_.gpu else att_org
    att_org = att_org.type(torch.float)
    att_list = [att_org]

    for i in range(args.n_attrs):
        tmp = att_org.clone()
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        att_list.append(tmp)

    if args_.use_model == 'generator':
        with torch.no_grad():
            samples = [img_real]
            for i, att_tar in enumerate():
                if i > 0:
                    att_diff = (att_tar - att_org) * args_.test_int
                    samples.append(cafegan.G(img_real, att_diff))
            samples = torch.cat(samples, dim=3)
            out_file = '{:06d}.jpg'.format(idx)
            vutils.save_image(
                samples, join(output_path, out_file),
                nrow=1, normalize=True, range=(-1., 1.)
            )
            print('{:s} done!'.format(out_file))

    elif args_.use_model == 'discriminator':
        _, mc, mw, mh = img_real.shape
        img_unit = img_real.clone().squeeze(0)
        img_unit = ((img_unit * 0.5) + 0.5) * 255
        img_unit = np.uint8(img_unit.cpu())
        img_unit = img_unit[::-1,:,:].transpose(1,2,0)
        with torch.no_grad():
            result1 = img_unit.copy()
            result2 = np.zeros_like(img_unit, dtype=np.uint8)
            abn_att_real, cabn_att_real = cafegan.D(img_real, 'att')
            abn_att = F.interpolate(abn_att_real, size=(mw, mh), mode='bilinear', align_corners=True).squeeze(0)
            cabn_att = F.interpolate(cabn_att_real, size=(mw, mh), mode='bilinear', align_corners=True).squeeze(0)
            for i in range(len(att_list)-1):
                abn_att_temp = np.uint8(abn_att[i].cpu() * 255)
                cabn_att_temp = np.uint8(cabn_att[i].cpu() * 255)
                heatmap_abn = cv2.applyColorMap(abn_att_temp, cv2.COLORMAP_JET)
                heatmap_cabn = cv2.applyColorMap(cabn_att_temp, cv2.COLORMAP_JET)
                result_abn = (heatmap_abn * 0.3 + img_unit * 0.5)
                result_cabn = (heatmap_cabn * 0.3 + img_unit * 0.5)

                result1 = np.append(result1, result_abn, axis=1)
                result2 = np.append(result2, result_cabn, axis=1)
            out_file = 'atts{:06d}.jpg'.format(idx)
            cv2.imwrite(join(output_path, out_file), np.concatenate((result1, result2), axis=0))
            print('{:s} done!'.format(out_file))