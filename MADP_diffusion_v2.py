import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

from MADP_Unet_v2 import TrajectoryUNet
from MADP_Unet_v2 import AxialPreprocessor, ContextEncoder, SinusoidalPosEmb, NoiseScheduler, BoundedNoiseScheduler

class MultiAgentDiffusionModel(nn.Module):
    def __init__(self,
                 max_agents     =10,
                 horizon        =20,
                 state_dim      =2,
                 img_ch         =3,
                 hid            =128,
                 diffusion_steps=1000,
                 t_min          =50):
        super().__init__()
        self.max_agents      = max_agents
        self.horizon         = horizon
        self.state_dim       = state_dim
        self.scheduler       = BoundedNoiseScheduler(diffusion_steps)
        self.time_emb        = SinusoidalPosEmb(hid)
        self.ctx_enc         = ContextEncoder(img_ch, pose_dim=2, hid=hid, max_agents=max_agents)
        self.preprocessor    = AxialPreprocessor(feat_dim=hid,
                                                 state_dim=2, 
                                                 num_heads=4,
                                                 max_agents=max_agents, 
                                                 T=horizon)
        self.unet            = TrajectoryUNet(max_agents, 
                                              horizon,
                                              state_dim, 
                                              feat_dim=hid, 
                                              hid_dim=hid,
                                              ctx_dim=hid)
        self.t_min           = t_min
        self.T_steps         = diffusion_steps

    def forward(self, frames, starts, goals, n_agents, trajs, active_agents):
        """
        frames: [B,3,H,W]
        starts: [B,maxA,2]
        goals:  [B,maxA,2]
        n_agents: [B]
        trajs:  [B,maxA,T,2]  padded
        active_agents: int   (for curriculum depth)
        """
        B = frames.size(0)
        device = frames.device

        # 1) context
        ctx = self.ctx_enc(frames, starts, goals, n_agents)
        

        # 2) choose t per-sample based on active_agents
        t_hi = int(self.t_min + (self.T_steps-self.t_min)*(active_agents/self.max_agents))
        t   = torch.randint(self.t_min, t_hi+1, (B,), device=device)

        # 3) forward noise
        x0 = trajs.permute(0,1,3,2)  # [B,maxA,2,T]
        x_noisy, noise = self.scheduler.q_sample(x0, t)
        x_noisy = x_noisy.permute(0,1,3,2)  # [B,maxA,T,2]
        

        # 4) build agent mask
        agent_mask = torch.arange(self.max_agents, device=device)[None,:] >= n_agents[:,None]

        # 5) axial attention + MLP mixer
        feat = self.preprocessor(x_noisy, agent_mask)  # [B,maxA,T,hid]

        # 6) time embed + fuse
        temb = self.time_emb(t)                        # [B,hid]
        cond = ctx + temb

        # 7) denoise
        pred_noise = self.unet(feat, cond)             # [B,maxA,T,2]


        # pred_noise: [B, Na, T, 2]
        # noise:      [B, Na, 2, T]  (after .permute(0,1,3,2))
        # agent_mask: [B, Na]  (True for padding slots)

        # 8) build a boolean mask of shape [B, Na, T, 2]
        mask = (~agent_mask)[:, :, None, None]  \
            .expand(-1, -1, pred_noise.size(2), pred_noise.size(3))

        # 9) select only the valid (non-padding) entries
        pn = pred_noise.masked_select(mask)
        nx = noise.permute(0,1,3,2).masked_select(mask)

        # 10) compute the MSE over those entries
        loss = F.mse_loss(pn, nx)
        
        return loss

    @torch.no_grad()
    def sample(self, frames, starts, goals, n_agents):
        """
        Reverse diffusion to generate trajectories
        Returns: [B,maxA,T,2] padded; user should slice [:,:n_agents]
        """
        B = frames.size(0)
        device = frames.device

        # 1) context
        ctx = self.ctx_enc(frames, starts, goals, n_agents)

        # 2) initialize noise
        x = torch.randn(B, self.max_agents, self.horizon, 2, device=device)

        # 3) reverse loop
        for i in reversed(range(self.T_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            # time embed + fuse
            temb = self.time_emb(t)
            cond = ctx + temb

            # agent mask
            agent_mask = torch.arange(self.max_agents,device=device)[None,:] >= n_agents[:,None]
            feat = self.preprocessor(x, agent_mask)

            eps = self.unet(feat, cond)                 # [B,maxA,T,2]
            beta = self.scheduler.betas[t].view(-1,1,1,1)
            a    = self.scheduler.alphas[t].view(-1,1,1,1)
            ac   = self.scheduler.alphas_cum[t].view(-1,1,1,1)

            # DDPM update
            mean = (x - ( (1-a)/torch.sqrt(1-ac) )*eps)/torch.sqrt(a)
            if i>0:
                x = mean + torch.sqrt(beta)*torch.randn_like(x)
            else:
                x = mean

        return x  # [B,maxA,T,2]
