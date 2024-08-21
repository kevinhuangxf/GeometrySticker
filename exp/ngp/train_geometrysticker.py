import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

# networks
import torch.nn.functional as F
from models.unet import VGG16Unet, VGG16UNetWM, WMDecoder
from models.decoder import Decoder_sigmoid, Decoder_sigmoid_classifier
# from models.hidden import HiddenDecoder

# optimizer, losses
# from apex.optimizers import FusedAdam
from torch.optim import AdamW as FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss
from lr_schedulers import CosineAnnealingRestartLR

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import slim_ckpt, load_ckpt, sharpen, GaussianSmoothing, gradient

import warnings; warnings.filterwarnings("ignore")
from kornia.color.yuv import rgb_to_yuv, yuv420_to_rgb
from kornia.color import rgb_to_grayscale

# import clip
# import yaml
from clip_utils import CLIPEditor

import torchvision.transforms as T
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import adjust_hue

color_jitter = T.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.5)


def rgb2gray(x):
    curf = 0.299 * x[:, [0], :, :] + 0.587 * x[:, [1], :, :] + 0.114 * x[:, [2], :, :]
    return curf

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # self.hparams['code_length'] = 64
        # self.hparams['use_code_decoder'] = False

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act, code_length=self.hparams.code_length)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

        self.model_origin = NGP(scale=self.hparams.scale, rgb_act=rgb_act, code_length=self.hparams.code_length)
        # self.model_origin = NGP(scale=self.hparams.scale, rgb_act=rgb_act, code_length=32)
        G = self.model_origin.grid_size
        self.model_origin.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model_origin.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

        if self.hparams.use_code_decoder:
            self.code_decoder = Decoder_sigmoid(decoder_channels=64, decoder_blocks=6, message_length=self.hparams.code_length)
            # self.code_decoder = Decoder_sigmoid_classifier(in_channels=4, decoder_channels=64, decoder_blocks=6, message_length=32)
        self.unet = VGG16Unet(pretrained=True)
        self.unet_wm = VGG16UNetWM(in_channel=4, out_channel=3, pretrained=True)
        self.wm_classifier = WMDecoder(in_channel=3, decoder_fc_out=1) # self.model.code.shape[1]
        self.wm_decoder = WMDecoder(in_channel=3, decoder_fc_out=self.model.code.shape[1])
        # self.hidden_decoder = HiddenDecoder(num_blocks=8, num_bits=self.model.code.shape[1], channels=3)
        self.label_en = GaussianSmoothing(1, 5, 1)
        self.criterion = nn.L1Loss(reduction='mean')
        self.clip_editor = CLIPEditor()
        self.clip_editor.text_features = self.clip_editor.encode_text([self.hparams.clip_prompt])
        self.wm_acc = 0 
        self.dis_wm_acc = 0

    def forward(self, batch, split, use_origin_model=False, use_wm=False):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']
        
        kwargs['use_origin_model'] = use_origin_model
        kwargs['use_wm'] = use_wm
        if use_origin_model:
            return render(self.model_origin, rays_o, rays_d, **kwargs)
        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.train_dataset_full_image = dataset(split='train', **kwargs)
        self.train_dataset_full_image.batch_size = self.train_dataset_full_image.img_wh[0] * self.train_dataset_full_image.img_wh[1]
        self.train_dataset_full_image.ray_sampling_strategy = 'same_image'

        self.test_dataset = dataset(split='test', **kwargs)

        print(len(self.test_dataset))

        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

    def configure_optimizers(self):
        # define additional parameters
        # self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        # self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)
        load_ckpt(self.model_origin, self.hparams.weight_path)

        net_params = []
        # for n, p in self.named_parameters():
        #     if n not in ['dR', 'dT']: net_params += [p]
            
        for n, p in self.model.named_parameters():
            # if n not in ['dR', 'dT']: net_params += [p]
            if n not in ['rgb_net.params', 'rgb_act.params', 'beta']: 
                net_params += [p]
                print(n)
        
        # for n, p in self.model.rgb_net.named_parameters():
        #     if n not in ['dR', 'dT']: net_params += [p]

        # for n, p in self.unet.named_parameters():
        #     if n not in ['dR', 'dT']: net_params += [p]

        # for n, p in self.model.code_mlp.named_parameters():
        #     if n not in ['dR', 'dT']: net_params += [p]
        
        # for n, p in self.model.wm_net.named_parameters():
        #     print(f'{n}: {p.shape}')
        #     if n not in ['dR', 'dT']: net_params += [p]

        # for n, p in self.wm_decoder.named_parameters():
        #     print(f'{n}: {p.shape}')
        #     if n not in ['dR', 'dT']: net_params += [p]

        # for n, p in self.unet_wm.named_parameters():
        #     print(f'{n}: {p.shape}')
        #     if n not in ['dR', 'dT']: net_params += [p]

        # for name, param in self.model.named_parameters():
        #     if name == "code_mlp.params":
        #         print(f'Attribute {name}: {param.shape} remain grad!')
        #         param.requires_grad = True
        #     else:
        #         print(f'Attribute {name}: {param.shape} close grad!')
        #         param.requires_grad = False

        net_params_beta = [self.model.beta]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        self.net_opt_beta = FusedAdam(net_params_beta, 0.01, eps=1e-15)

        opts += [self.net_opt]
        opts += [self.net_opt_beta]

        # code decoder
        if self.hparams.use_code_decoder:
            code_params = []
            for n, p in self.code_decoder.named_parameters():
                print(f'{n}: {p.shape}')
                if n not in ['dR', 'dT']: code_params += [p]

            code_opt = FusedAdam(code_params, 1e-4, eps=1e-15)
            opts += [code_opt]

        # wm 
        wm_params = []
        for n, p in self.wm_classifier.named_parameters():
            print(f'{n}: {p.shape}')
            if n not in ['dR', 'dT']: wm_params += [p]
        for n, p in self.wm_decoder.named_parameters():
            print(f'{n}: {p.shape}')
            if n not in ['dR', 'dT']: wm_params += [p]
        # for n, p in self.hidden_decoder.named_parameters():
        #     print(f'{n}: {p.shape}')
        #     if n not in ['dR', 'dT']: wm_params += [p]

        wm_opt = FusedAdam(wm_params, 1e-6, eps=1e-15)
        opts += [wm_opt]

        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/10)

        # net_sch = CosineAnnealingRestartLR(self.net_opt,
        #                                    periods=[30, 30],
        #                                    restart_weights=[1, 0.5],
        #                                    eta_min=1e-5)

        net_sch = CosineAnnealingLR(self.net_opt,
                            self.hparams.num_epochs,
                            self.hparams.lr/10)

        return opts, [net_sch]

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)
        train_full_image_dataloader = DataLoader(self.train_dataset_full_image,
                                                 num_workers=16,
                                                 persistent_workers=True,
                                                 batch_size=None,
                                                 pin_memory=True)
        return [train_dataloader, train_full_image_dataloader]

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)
        
        # for name, param in self.named_parameters():
        #     if name == "model.code_mlp.params":
        #         print(f'Attribute {name}: {param.shape} remain grad!')
        #         param.requires_grad = True
        #     else:
        #         print(f'Attribute {name}: {param.shape} close grad!')
        #         param.requires_grad = False

    def train_nerf(self, batch):
        # render train        
        # results = self(batch, split='train')
        results = self(batch, split='train', use_origin_model=False, use_wm=True)
        loss_d = self.loss(results, batch)
        if self.hparams.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss_render', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), False)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), False)
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def train_wm(self, batch):
        # render test
        with torch.no_grad():
            results = self(batch, split='train', use_origin_model=False, use_wm=True)

        with torch.no_grad():
            results_origin = self(batch, split='train', use_origin_model=True, use_wm=False)

        w, h = self.train_dataset.img_wh
        rgb_img = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_img_origin = rearrange(results_origin['rgb'], '(h w) c -> 1 c h w', h=h)

        # sharpen
        # gray_img = rgb_to_grayscale(rgb_img)
        # gray_img_origin = rgb_to_grayscale(rgb_img_origin)
        # gray_img_origin_sharp = sharpen(gray_img_origin, self.label_en, LABEL_SHARP=1.0)
        # gray_img_origin_sharp = gray_img_origin_sharp.clamp_(*(0, 1))

        if self.global_step % self.trainer.log_every_n_steps == 0:
            self.trainer.logger.experiment.add_images(f'rgb_img', rgb_img, self.global_step)

        x_ = torch.cat((rgb_img, rgb_img_origin), 0)
        # x_ = torch.cat((gray_img, gray_img_origin_sharp), 0)
        # x_out, x_c = self.unet_wm(x_, wm_resize=self.wm_size)
        x_dec, x_c = self.wm_decoder(x_)
        _, x_c = self.wm_classifier(x_)
        # x_dec, _ = self.wm_decoder(x_)
        # x_dec = self.hidden_decoder(x_)
        out_c, out_c_ctr = x_c[:1], x_c[1:]
        out_wm, out_wm_ctr = x_dec[:1], x_dec[1:]
        wm_pred, ctr_pred = (out_wm >= 0.5).float(), (out_wm < 0.5).float()

        # # wm loss
        bs = 1
        dis_wm_loss = nn.BCEWithLogitsLoss()(x_c, torch.stack( [torch.ones(bs), torch.zeros(bs)]).to(x_c.device) )
        wm_loss = nn.BCEWithLogitsLoss()(out_wm, self.model.code)
        # wm_ctr_loss = nn.BCEWithLogitsLoss()(out_wm_ctr, torch.zeros_like(out_wm_ctr))

        # decode
        if self.hparams.use_code_decoder:
            rgb_decode = self.code_decoder(rgb_img, out_c)
            code_loss = nn.BCEWithLogitsLoss()(rgb_decode, self.model.code)
            code_pred = (rgb_decode >= 0.5).float()
            code_acc = torch.sum(code_pred == self.model.code) / self.model.code.size(1)

        # Compute the accuracy
        wm_acc = torch.sum(wm_pred == self.model.code) / self.model.code.size(1)
        dis_wm_acc = ( torch.sum(out_c>=0.5) + torch.sum(out_c_ctr<0.5) ) / ( torch.sum(torch.ones_like(out_c)) + torch.sum(torch.ones_like(out_c_ctr)) ) 
        self.wm_acc = wm_acc
        self.dis_wm_acc = dis_wm_acc

        loss_wm = dis_wm_loss + wm_loss # + wm_ctr_loss
        if self.hparams.use_code_decoder:
            loss_wm += code_loss
        loss_wm = loss_wm.mean()

        self.log('train/dis_wm_loss', dis_wm_loss, True)
        self.log('train/wm_loss', wm_loss, True)
        # self.log('train/wm_ctr_loss', wm_ctr_loss)
        self.log('train/dis_wm_acc', dis_wm_acc)
        self.log('train/wm_acc', wm_acc)
        if self.hparams.use_code_decoder:
            self.log('train/code_loss', code_loss, True)
            self.log('train/code_acc', code_acc, True)

        loss = loss_wm

        # if self.global_step == 1:
        #     self.trainer.logger.experiment.add_images(f'style_img', self.style_img, self.global_step)
        # self.trainer.logger.experiment.add_images(f'wm/ wm_ctr', torch.cat((out_wm, out_wm_ctr), -1), self.global_step)
        if self.global_step % self.trainer.log_every_n_steps == 0:
            self.trainer.logger.experiment.add_images(f'rgb_img/ rgb_img_origin/ diff', torch.cat((rgb_img, rgb_img_origin, (rgb_img - rgb_img_origin)), -1), self.global_step)
            # self.trainer.logger.experiment.add_images(f'rgb_img_origin/ rgb_img_origin_sharp/ diff', torch.cat((gray_img_origin, gray_img_origin2, (gray_img_origin2 - gray_img_origin)), -1), self.global_step)
            # self.trainer.logger.experiment.add_images(f'gray_img/ gray_img_origin_sharp/ diff', torch.cat((gray_img, gray_img_origin_sharp, (gray_img_origin_sharp - gray_img)), -1), self.global_step)
        # self.trainer.logger.experiment.add_images(f'depth_pred/ depth_pred_origin/ diff', torch.cat((depth_pred, depth_pred_origin, (depth_pred - depth_pred_origin)), -1), self.global_step)
        
        return loss
    
    def train_wm_image(self, batch):

        with torch.no_grad():
            results = self(batch, split='train')

        with torch.no_grad():
            results_origin = self(batch, split='train', use_origin_model=True)

        w, h = self.train_dataset.img_wh
        rgb_img = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_img_origin = rearrange(results_origin['rgb'], '(h w) c -> 1 c h w', h=h)

        x_ = torch.cat((rgb_img, rgb_img_origin), 0)

        _, x_c = self.wm_decoder(x_)
        out_c, out_c_ctr = x_c[:1], x_c[1:]

        x_out, _ = self.unet_wm(x_, x_c, wm_resize=self.wm_size)
        out_wm, out_wm_ctr = x_out[:1], x_out[1:]

        # wm loss
        bs = 1
        dis_wm_loss = nn.BCEWithLogitsLoss()(x_c, torch.stack( [torch.ones(bs), torch.zeros(bs)]).to(x_out.device) )
        dis_wm_acc = ( torch.sum(out_c>=0.5) + torch.sum(out_c_ctr<0.5) ) / ( torch.sum(torch.ones_like(out_c)) + torch.sum(torch.ones_like(out_c_ctr)) ) 
        wm_loss = F.mse_loss(out_wm, self.style_img)
        wm_ctr_loss = F.mse_loss(out_wm_ctr, torch.zeros_like(out_wm_ctr))

        loss_wm = 3.0 * dis_wm_loss + wm_loss + wm_ctr_loss
        loss_wm = loss_wm.mean()

        self.log('train/loss_wm', loss_wm)
        self.log('train/dis_wm_loss', dis_wm_loss)
        self.log('train/dis_wm_acc', dis_wm_acc)
        self.log('train/wm_loss', wm_loss)
        self.log('train/wm_ctr_loss', wm_ctr_loss)

        if self.global_step == 1:
            self.trainer.logger.experiment.add_images(f'style_img', self.style_img, self.global_step)
        if self.global_step % self.trainer.log_every_n_steps == 0:
            self.trainer.logger.experiment.add_images(f'wm/ wm_ctr', torch.cat((out_wm, out_wm_ctr), -1), self.global_step)

        return loss_wm
    
    def train_clip(self, batch):

        results = self(batch, split='train', use_origin_model=False, use_wm=True)

        w, h = self.train_dataset.img_wh
        rgb_img = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)

        # make rendered patch (with/without augmentations) similar to target text via clip
        sample_N_aug = 5  # N random augmentations
        clip_emb = self.clip_editor.encode_image(rgb_img, preprocess=True, stochastic=sample_N_aug)  # (N_aug, dim)
        clip_loss = 1.0 - (self.clip_editor.text_features.float()[None] * clip_emb).sum(dim=-1)
        self.log('train/clip_loss', clip_loss.mean())
        loss = clip_loss.mean()        

        if self.global_step % self.trainer.log_every_n_steps == 0:
            self.trainer.logger.experiment.add_images(f'rgb_clip_img', rgb_img, self.global_step)
        
        with torch.no_grad():
            x_dec, x_c = self.wm_decoder(rgb_img)
            self.log('train/clip_wm', x_c)
            wm_pred = (x_dec >= 0.5).float()
            wm_acc = torch.sum(wm_pred == self.model.code) / self.model.code.size(1)
            self.log('train/clip_wm_acc', wm_acc)

        return loss

    def training_step(self, batch, batch_nb, *args):
        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=self.hparams.dataset_name=='colmap')

        batch_train, batch_train_full_image = batch

        # if self.current_epoch < 1:
        # if self.global_step < 10:
            # loss_nerf = self.train_nerf(batch_train)
            # loss_nerf = self.train_nerf(batch_train_full_image)
            # loss = 1.0 * loss_nerf

        # update random code
        if self.hparams.random_code:
            self.model.update_code()



        loss = 0
        # train wm
        loss_nerf = self.train_nerf(batch_train_full_image)
        loss_wm = self.train_wm(batch_train_full_image)
        loss += 1.0 * loss_nerf + 1.0 * loss_wm

        # train clip
        # if self.current_epoch <= 2:
        #     # loss_nerf = self.train_nerf(batch_train)
        #     # loss_wm = self.train_wm_image(batch_train_full_image)
        #     loss_nerf = self.train_nerf(batch_train_full_image)
        #     loss_wm = self.train_wm(batch_train_full_image)
        #     loss += 1.0 * loss_nerf + 1.0 * loss_wm
        # elif self.wm_acc > 0.9 or self.dis_wm_acc == 1:
        #     loss_clip = self.train_clip(batch_train_full_image)
        #     loss += 1.0 * loss_clip
        # else:
        #     loss_wm = self.train_wm(batch_train_full_image)
        #     loss += 1.0 * loss_wm

        # loss = self.train_wm(batch_train_full_image)
        # loss = self.train_nerf_sharp(batch_train_full_image)
        # loss = self.train_nerf(batch_train)

        self.log('lr', self.net_opt.param_groups[0]['lr'])

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        # results = self(batch, split='test')
        # use_origin_model=False, use_wm=False # default
        results = self(batch, split='test', use_origin_model=False, use_wm=True)
        results_origin = self(batch, split='test', use_origin_model=True, use_wm=False)

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_origin = rearrange(results_origin['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                           torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()
        
        x_dec, x_c = self.wm_decoder(rgb_pred)
        wm_pred = (x_dec >= 0.5).float()
        wm_acc = torch.sum(wm_pred == self.model.code) / self.model.code.size(1)
        logs['wm_acc'] = wm_acc
        # print("decode wm_acc: ", wm_acc)
        
        _, x_c = self.wm_classifier(rgb_pred)
        # x_dec = self.hidden_decoder(x_)
        out_c, out_c_ctr = x_c[:1], x_c[1:]
        dis_wm_acc = ( torch.sum(out_c>=0.5) + torch.sum(out_c_ctr<0.5) ) / ( torch.sum(torch.ones_like(out_c)) + torch.sum(torch.ones_like(out_c_ctr)) ) 
        logs['dis_wm_acc'] = dis_wm_acc

        # code decode
        if self.hparams.use_code_decoder:
            rgb_decode = self.code_decoder(rgb_pred, x_c)
            code_pred = (rgb_decode >= 0.5).float()
            code_acc = torch.sum(code_pred == self.model.code) / self.model.code.size(1)
            logs['code_acc'] = code_acc

        rgb_pred_jitter = color_jitter(rgb_pred)
        x_dec_jitter, x_c_jitter = self.wm_decoder(rgb_pred_jitter)
        wm_pred_jitter = (x_dec_jitter >= 0.5).float()
        wm_acc_jitter = torch.sum(wm_pred_jitter == self.model.code) / self.model.code.size(1)
        logs['wm_acc_jitter'] = wm_acc_jitter
        # print("jitter decode wm_acc: ", wm_acc_jitter)

        _, x_c_jitter = self.wm_classifier(rgb_pred_jitter)
        # x_dec = self.hidden_decoder(x_)
        out_c_jitter, out_c_ctr_jitter = x_c_jitter[:1], x_c_jitter[1:]
        dis_wm_acc_jitter = ( torch.sum(out_c_jitter>=0.5) + torch.sum(out_c_ctr_jitter<0.5) ) / ( torch.sum(torch.ones_like(out_c_jitter)) + torch.sum(torch.ones_like(out_c_ctr_jitter)) ) 
        logs['dis_wm_acc_jitter'] = dis_wm_acc_jitter

        # code decode jittor
        if self.hparams.use_code_decoder:
            rgb_decode = self.code_decoder(rgb_pred_jitter, x_c_jitter)
            code_pred_jitter = (rgb_decode >= 0.5).float()
            code_acc_jitter = torch.sum(code_pred_jitter == self.model.code) / self.model.code.size(1)
            logs['code_acc_jitter'] = code_acc_jitter

        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            rgb_origin = rearrange(results_origin['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_origin = (rgb_origin*255).astype(np.uint8)
            rgb_pred_jitter_img = rearrange(rgb_pred_jitter[0].cpu().numpy(), 'c h w -> h w c', h=h)
            rgb_pred_jitter_img = (rgb_pred_jitter_img*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            # imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
            # imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_o.png'), rgb_origin)
            # imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_diff.png'), np.abs(rgb_pred-rgb_origin))
            
            # background_mask
            bg_mask = (rgb_gt == 1.0)
            bg_mask = rearrange(bg_mask.cpu().numpy(), '1 c h w -> h w c', h=h)
            # bg_mask = (bg_mask*255).astype(np.uint8)

            def change_hue(img, factor = 0.5):
                # Assuming rgb_pred is your image tensor
                # Convert tensor image to PIL Image
                to_pil = ToPILImage()
                rgb_pred_pil = to_pil(img)

                # adjust the t=hue
                rgb_pred_hue_adjusted_pil = adjust_hue(rgb_pred_pil, factor)

                return np.array(rgb_pred_hue_adjusted_pil)
            
            rgb_pred_hue = change_hue(rgb_pred, factor=-0.25)
            rgb_origin_hue = change_hue(rgb_origin, factor=-0.25)

            diff_image = np.abs(rgb_pred - rgb_origin)
            diff_image[bg_mask] = 0

            diff_image_hue = np.abs(rgb_pred_hue-rgb_origin_hue)
            diff_image_hue[bg_mask] = 0

            # Concatenate the images side by side
            # concatenated_img = np.concatenate((rgb_origin, rgb_pred, np.abs(rgb_pred-rgb_origin)), axis=1)
            # concatenated_img_hue = np.concatenate((rgb_origin_hue, rgb_pred_hue, np.abs(rgb_pred_hue-rgb_origin_hue)), axis=1)
            concatenated_img = np.concatenate((rgb_origin, rgb_pred, diff_image), axis=1)
            concatenated_img_hue = np.concatenate((rgb_origin_hue, rgb_pred_hue, diff_image_hue), axis=1)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_concatenated.png'), concatenated_img)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}__concatenated_hue.png'), concatenated_img_hue)

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)
        
        wm_acc = torch.stack([x['wm_acc'] for x in outputs])
        mean_wm_acc = all_gather_ddp_if_available(wm_acc).mean()
        self.log('test/wm_acc', mean_wm_acc, True)
        print("mean_wm_acc: ", mean_wm_acc)

        dis_wm_acc = torch.stack([x['dis_wm_acc'] for x in outputs])
        mean_dis_wm_acc = all_gather_ddp_if_available(dis_wm_acc).mean()
        self.log('test/mean_dis_wm_acc', mean_dis_wm_acc, True)
        print("mean_dis_wm_acc: ", mean_dis_wm_acc)

        wm_acc_jitter = torch.stack([x['wm_acc_jitter'] for x in outputs])
        mean_wm_acc_jitter = all_gather_ddp_if_available(wm_acc_jitter).mean()
        self.log('test/wm_acc_jitter', mean_wm_acc_jitter)
        print("mean_wm_acc_jitter: ", mean_wm_acc_jitter)

        dis_wm_acc_jitter = torch.stack([x['dis_wm_acc_jitter'] for x in outputs])
        mean_dis_wm_acc_jitter = all_gather_ddp_if_available(dis_wm_acc_jitter).mean()
        self.log('test/mean_dis_wm_acc_jitter', mean_dis_wm_acc_jitter)
        print("mean_dis_wm_acc_jitter: ", mean_dis_wm_acc_jitter)

        if self.hparams.use_code_decoder:
            code_acc = torch.stack([x['code_acc'] for x in outputs])
            mean_code_acc = all_gather_ddp_if_available(code_acc).mean()
            self.log('test/code_acc', mean_code_acc, True)
            print("mean_code_acc: ", mean_code_acc)

            code_acc_jitter = torch.stack([x['code_acc_jitter'] for x in outputs])
            mean_code_acc_jitter = all_gather_ddp_if_available(code_acc_jitter).mean()
            self.log('test/mean_code_acc_jitter', mean_code_acc_jitter)
            print("mean_code_acc_jitter: ", mean_code_acc_jitter)

        # torch.save(self.wm_decoder.state_dict(), '/home/comp/csxfhuang/development/ngp_pl/wm_decoder.pth')


    # def get_progress_bar_dict(self):
    #     # don't show the version number
    #     items = super().get_progress_bar_dict()
    #     items.pop("v_num", None)
    #     return items


if __name__ == '__main__':
    hparams = get_opts()
    # if hparams.val_only and (not hparams.ckpt_path):
    #     raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    print("hparams.limit_train_batches: ", hparams.limit_train_batches)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=1,#hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16,
                      limit_train_batches=hparams.limit_train_batches,
                      limit_val_batches=200
                    )

    # trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if hparams.val_only:
        # hparams_ = system.hparams.code_length
        # print(hparams_)
        if hparams.ckpt_path is not None:
            system.load_from_checkpoint(hparams.ckpt_path, strict=True)
            system.hparams['code_length'] = system.hparams.code_length
        if hparams.weight_path is not None:
            load_ckpt(system.model, system.hparams.weight_path, model_name="model", prefixes_to_ignore=['origin'])
            load_ckpt(system.model_origin, system.hparams.weight_path, model_name="model_origin")
        trainer.validate(system)
    else:
        trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim_geosticker.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name=='nsvf' and \
       'Synthetic' in hparams.root_dir: # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'recolor.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'origin.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)
