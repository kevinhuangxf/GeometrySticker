import torch
from torch import nn
import tinycudann as tcnn
import vren
from einops import rearrange
from .custom_functions import TruncExp
import numpy as np

from .rendering import NEAR_DISTANCE
from models.density import LaplaceDensity

N = 16384 * 10

class NGP(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid', code_length=32):
        super().__init__()

        self.rgb_act = rgb_act

        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
            torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=16,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        self.rgb_net = \
            tcnn.Network(
                n_input_dims=32, n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": self.rgb_act,
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )

        if self.rgb_act == 'None': # rgb_net output is log-radiance
            for i in range(3): # independent tonemappers for r,g,b
                tonemapper_net = \
                    tcnn.Network(
                        n_input_dims=1, n_output_dims=1,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "Sigmoid",
                            "n_neurons": 64,
                            "n_hidden_layers": 1,
                        }
                    )
                setattr(self, f'tonemapper_net_{i}', tonemapper_net)

        self.code = torch.randint(0, 2, (1, code_length)).half().cuda()
        self.code_length = code_length
        self.code_mlp = \
            tcnn.Network(
                n_input_dims=self.code.shape[1], n_output_dims=32,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 32,
                    "n_hidden_layers": 1,
                }
            )
        # self.wm_net = \
        #     tcnn.Network(
        #         n_input_dims=64, n_output_dims=32,
        #         network_config={
        #             "otype": "FullyFusedMLP",
        #             "activation": "ReLU",
        #             "output_activation": "ReLU",
        #             "n_neurons": 64,
        #             "n_hidden_layers": 1,
        #         }
        #     )

        layer1 = torch.nn.Linear(self.code_length * 2, 64)
        # layer1 = torch.nn.Linear(self.code_length, 64)
        layer2 = torch.nn.Linear(64, 64)
        layer3 = torch.nn.Linear(64, self.code_length)
        # torch.nn.init.constant_(layer3.bias, 0)
        self.wm_net = torch.nn.Sequential(layer1, torch.nn.LeakyReLU(inplace=True),
                                       layer2, torch.nn.LeakyReLU(inplace=True),
                                       layer3)

        self.laplace_density = LaplaceDensity(dict(beta=0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def update_code(self):
        self.code = torch.randint(0, 2, (1, self.code_length)).half().cuda()

    def density(self, x, return_feat=False, kwargs={}):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        h = self.xyz_encoder(x)

        # if 'use_wm' in kwargs.keys():
        #     if kwargs['use_wm']:
        #         large_value_list = torch.argsort(h[:, 0], descending=True) # sigmas
        #         large_value_list_N = large_value_list # large_value_list[:N]

        #         large_mod = len(large_value_list_N) % 32
        #         if large_mod != 0:
        #             large_value_list_N = large_value_list_N[large_mod:]

        #         largest_N_values = torch.index_select(h[:, 0], 0, large_value_list_N) # sigmas

        #         largest_N_values = largest_N_values.view(-1, 32)
        #         code_ = self.code_mlp(self.code)
        #         code_ = code_.repeat(largest_N_values.shape[0], 1)
        #         largest_N_values_wm = self.wm_net(torch.cat([largest_N_values, code_], dim=1))
        #         # sigmas[large_value_list_N] = TruncExp.apply(largest_N_values_wm.view(-1))
        #         h[:, 0][large_value_list_N] = largest_N_values_wm.view(-1)
        #         # sigmas = TruncExp.apply(h[:, 0])

        #         # h[:, 0] = torch.clamp(h[:, 0], min=0)
        #         # h[:, 0] = torch.sign(h[:, 0]) * h[:, 0]

        # s = sum(h[:, 0].lt(0)) / len(h[:, 0])

        sigmas = TruncExp.apply(h[:, 0])

        if 'use_wm' in kwargs.keys():

            def lap_cdf(x, beta):
                return 0.5 - 0.5 * (x-x.mean()).sign() * torch.expm1(-(x-x.mean()).abs() / beta)

            if kwargs['use_wm'] and len(sigmas) > 5000:
                laplace_cdf = lambda x, beta: 0.5 - 0.5 * (x-x.mean()).sign() * torch.expm1(-(x-x.mean()).abs() / beta)

                # sigma_cdf = laplace_cdf(sigmas, 200)
                # sigma_cdf = laplace_cdf(sigmas, self.beta)
                sigma_cdf = lap_cdf(sigmas, self.beta)
                large_idx = torch.where(sigma_cdf > 0.999, 1, 0).nonzero().squeeze(1)
                large_idx = torch.from_numpy(np.array(list(range(len(sigmas))))).cuda()
                # large_idx = (sigma_cdf > 0.999).nonzero().squeeze(1)
                # large_idx = torch.where(sigmas > 3000, 1, 0).nonzero().squeeze(1)

                large_mod = len(large_idx) % 32
                if large_mod != 0:
                    large_idx = large_idx[large_mod:]
                # large_values = torch.index_select(sigmas, 0, large_idx)
                large_values = torch.index_select(h[:, 0], 0, large_idx)
                large_values = large_values.view(-1, 32)

                code_ = self.code_mlp(self.code)
                code_ = code_.repeat(large_values.shape[0], 1)
                large_values_wm = self.wm_net(torch.cat([large_values, code_], dim=1))
                # wm_values = self.wm_net(code_)
                # large_values_wm = large_values + wm_values
                
                h_clone = h[:, 0].clone()
                h_clone[large_idx] = large_values_wm.view(-1) * self.beta
                sigmas = TruncExp.apply(h_clone)

                # sigmas[large_idx] = large_values_wm.view(-1).float()
                # h[:, 0][large_idx] = large_values_wm.view(-1)
                # sigmas = TruncExp.apply(h[:, 0])

                h_new = torch.cat([h_clone.unsqueeze(1), h[:, 1:]], dim=1)

                # sigmas = self.beta * sigmas

                return sigmas, h_new

        # laplace density
        # sigmas = self.laplace_density(h[:, 0])
        
        if return_feat: return sigmas, h
        return sigmas

    def log_radiance_to_rgb(self, log_radiances, **kwargs):
        """
        Convert log-radiance to rgb as the setting in HDR-NeRF.
        Called only when self.rgb_act == 'None' (with exposure)

        Inputs:
            log_radiances: (N, 3)

        Outputs:
            rgbs: (N, 3)
        """
        if 'exposure' in kwargs:
            log_exposure = torch.log(kwargs['exposure'])
        else: # unit exposure by default
            log_exposure = 0

        out = []
        for i in range(3):
            inp = log_radiances[:, i:i+1]+log_exposure
            out += [getattr(self, f'tonemapper_net_{i}')(inp)]
        rgbs = torch.cat(out, 1)
        return rgbs

    def forward(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        if 'use_origin_model' not in kwargs:
            kwargs['use_origin_model'] = False
        sigmas, h = self.density(x, True, kwargs=kwargs)

        # if kwargs['use_wm']:
            
        #     large_value_list = torch.argsort(h[:, 0], descending=True) # sigmas
        #     large_value_list_N = large_value_list[:N]
        #     largest_N_values = torch.index_select(h[:, 0], 0, large_value_list_N) # sigmas
        #     largest_N_values = largest_N_values.view(-1, 32)
        #     code_ = self.code_mlp(self.code)
        #     code_ = code_.repeat(largest_N_values.shape[0], 1)
        #     largest_N_values_wm = self.wm_net(torch.cat([largest_N_values, code_], dim=1))
        #     # sigmas[large_value_list_N] = TruncExp.apply(largest_N_values_wm.view(-1))
        #     h[:, 0][large_value_list_N] = largest_N_values_wm.view(-1)
        #     sigmas = TruncExp.apply(h[:, 0])

        #     # result_values = largest_values + 10.0 * code_mlp_out
        #     # result_values = largest_values * code_mlp_out
        #     # h0_wm = self.code_mlp(torch.cat([h[:, 0:1], self.code.unsqueeze(0).repeat(h.shape[0], 1)]))

        #     # sigmas[large_value_list[:100000]] = 0
        #     # sigmas_mask = sigmas < sigmas[large_value_list[wm_size]]
        #     # sigmas_mask = sigmas > 10000
        #     # sigmas[sigmas_mask] = 0

        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d+1)/2)
        rgbs = self.rgb_net(torch.cat([d, h], 1))

        if self.rgb_act == 'None': # rgbs is log-radiance
            if kwargs.get('output_radiance', False): # output HDR map
                rgbs = TruncExp.apply(rgbs)
            else: # convert to LDR using tonemapper networks
                rgbs = self.log_radiance_to_rgb(rgbs, **kwargs)

        return sigmas, rgbs

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>density_threshold)[:, 0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a') # (N_cams, 3, 3)
        w2c_T = -w2c_R@poses[:, :3, 3:] # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i+chunk]/(self.grid_size-1)*2-1
                s = min(2**(c-1), self.scale)
                half_grid_size = s/self.grid_size
                xyzs_w = (xyzs*(s-half_grid_size)).T # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T # (N_cams, 3, chunk)
                uvd = K @ xyzs_c # (N_cams, 3, chunk)
                uv = uvd[:, :2]/uvd[:, 2:] # (N_cams, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2]>=NEAR_DISTANCE)&in_image # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i+chunk]] = \
                    count = covered_by_cam.sum(0)/N_cams

                too_near_to_cam = (uvd[:, 2]<NEAR_DISTANCE)&in_image # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count>0)&(~too_near_to_any_cam)
                self.density_grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup: # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c-1), self.scale)
            half_grid_size = s/self.grid_size
            xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid<0,
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid>0].mean().item()

        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)
