import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pdb

# -----------------------------------------------------------------------------
# NoiseScheduler & Positional Embedding
# -----------------------------------------------------------------------------
class NoiseScheduler(nn.Module):
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cum', alphas_cum)
        self.register_buffer('sqrt_ac', torch.sqrt(alphas_cum))
        self.register_buffer('sqrt_1mac', torch.sqrt(1 - alphas_cum))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a  = self.sqrt_ac[t].view(-1,1,1,1)
        am = self.sqrt_1mac[t].view(-1,1,1,1)
        return a*x0 + am*noise, noise
    
class BoundedNoiseScheduler(nn.Module):
    def __init__(self, T=1000, max_noise_ratio=3.0):
        super().__init__()
        # Cosine schedule with bounded noise
        steps = torch.arange(T)
        s = 0.008
        alphas_cum = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cum = alphas_cum / alphas_cum[0]
        
        # Constrain minimum signal retention
        min_alpha = 1.0 / (1.0 + max_noise_ratio**2)
        alphas_cum = torch.clamp(alphas_cum, min=min_alpha, max=1.0)
        
        betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
        betas = torch.clamp(betas, 0, 0.999)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('alphas_cum', alphas_cum)
        self.register_buffer('sqrt_ac', torch.sqrt(alphas_cum))
        self.register_buffer('sqrt_1mac', torch.sqrt(1 - alphas_cum))
    
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a  = self.sqrt_ac[t].view(-1,1,1,1)
        am = self.sqrt_1mac[t].view(-1,1,1,1)
        return a*x0 + am*noise, noise

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim=dim
    def forward(self, t):
        half = self.dim//2
        freqs = torch.exp(-math.log(10000)*(torch.arange(half,device=t.device)/(half-1)))
        args = t.unsqueeze(1).float()*freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

# -----------------------------------------------------------------------------
# Context Encoder: image + start/goal + agent count
# -----------------------------------------------------------------------------
class ContextEncoder(nn.Module):
    def __init__(self, img_ch=3, pose_dim=2, hid=128, max_agents=10):
        super().__init__()
        res = nn.Sequential(*list(__import__('torchvision').models.resnet18(pretrained=False).children())[:-2])
        self.img_enc = nn.Sequential(res, nn.AdaptiveAvgPool2d(1),
                                     nn.Flatten(), nn.Linear(512,hid), nn.ReLU())
        self.pose_enc= nn.Sequential(nn.Linear(2*max_agents*2, hid), nn.ReLU(), nn.Linear(hid,hid))
        self.count_emb= nn.Embedding(max_agents+1, hid)
        self.ctx_attn = nn.MultiheadAttention(hid,4, batch_first=True)

    def forward(self, img, start, goal, n_agents):
        B = img.size(0)
        f_img  = self.img_enc(img)
        poses  = torch.cat([start, goal], dim=-1).flatten(1)
        f_pose = self.pose_enc(poses)
        f_cnt  = self.count_emb(n_agents)
        tokens = torch.stack([f_img, f_pose, f_cnt], dim=1)  # [B,3,hid]
        attn,_ = self.ctx_attn(tokens, tokens, tokens)
        return attn.mean(1)  # [B,hid]

# -----------------------------------------------------------------------------
# Axial Preprocessor: agent-axis & time-axis attention + MLP mixer
# -----------------------------------------------------------------------------

class AxialPreprocessor(nn.Module):
    def __init__(self,
                 state_dim: int,    # originally 2 (x,y)
                 feat_dim:  int,    # e.g. 128
                 num_heads: int,    # e.g. 4
                 max_agents: int,
                 T:          int):
        super().__init__()
        self.max_agents = max_agents
        self.T          = T

        # 1) Input projection: from raw (x,y) → feat_dim
        self.input_proj = nn.Linear(state_dim, feat_dim)

        # 2) Agent-axis self-attention
        self.agent_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 3) Time-axis self-attention
        self.time_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 4) MLP mixer (keeps feat_dim in/out)
        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, x: torch.Tensor, agent_mask: torch.Tensor):
        """
        x:           [B, Na, T, state_dim]
        agent_mask:  [B, Na]  True for padding slots
        returns:     [B, Na, T, feat_dim]
        """
        B, Na, T, _ = x.shape

        # ————————————————————————————————————————————
        # 1) Project into hidden space
        # ————————————————————————————————————————————
        # x_proj: [B, Na, T, feat_dim]
        x_proj = self.input_proj(x)

        # ————————————————————————————————————————————
        # 2) Agent-axis attention
        #    reshape to [B*T, Na, feat_dim]
        # ————————————————————————————————————————————
        xa = x_proj.permute(0, 2, 1, 3).reshape(B * T, Na, -1)
        # expand mask to [B*T, Na]
        am = agent_mask.unsqueeze(1).expand(B, T, Na).reshape(B * T, Na)
        xa2, _ = self.agent_attn(xa, xa, xa, key_padding_mask=am)
        # restore shape: [B, Na, T, feat_dim]
        xa2 = xa2.view(B, T, Na, -1).permute(0, 2, 1, 3)

        # ————————————————————————————————————————————
        # 3) Time-axis attention
        #    reshape to [B*Na, T, feat_dim]
        # ————————————————————————————————————————————
        xt = xa2.reshape(B * Na, T, -1)
        xt2, _ = self.time_attn(xt, xt, xt)
        # back to [B, Na, T, feat_dim]
        xt2 = xt2.view(B, Na, T, -1)

        # ————————————————————————————————————————————
        # 4) MLP mixer
        #    maintains the same last‐dim feat_dim
        # ————————————————————————————————————————————
        out = self.mlp(xt2)  # [B, Na, T, feat_dim]
        return out


# -----------------------------------------------------------------------------
# U-Net Components
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# DownBlock: Conv1d downsampling + FiLM
# -----------------------------------------------------------------------------
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ctx_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act   = nn.SiLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.film  = nn.Linear(ctx_dim, out_ch * 2)

    def forward(self, x, ctx):
        # x: [B, in_ch, T] → downsampled to [B, out_ch, T/2]
        h = self.act(self.norm1(self.conv1(x)))
        # FiLM conditioning
        scale, shift = self.film(ctx).chunk(2, dim=-1)  # each [B, out_ch]
        
        # Fix broadcasting
        scale = scale.view(scale.size(0), scale.size(1), 1)
        shift = shift.view(shift.size(0), shift.size(1), 1)

        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.act(self.norm2(self.conv2(h)))
        return h

# -----------------------------------------------------------------------------
# MiddleBlock: bottleneck + temporal self-attention + FiLM
# -----------------------------------------------------------------------------
class MiddleBlock(nn.Module):
    def __init__(self, ch, ctx_dim, num_heads=4):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, ch)
        self.act   = nn.SiLU()
        self.film  = nn.Linear(ctx_dim, ch * 2)
        self.attn  = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)

    def forward(self, x, ctx):
        # x: [B, ch, T]
        h = self.act(self.norm1(self.conv1(x)))
        scale, shift = self.film(ctx).chunk(2, dim=-1)

        # Fix broadcasting
        scale = scale.view(scale.size(0), scale.size(1), 1)
        shift = shift.view(shift.size(0), shift.size(1), 1)

        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        # temporal self-attention
        h_t, _ = self.attn(h.permute(0,2,1), h.permute(0,2,1), h.permute(0,2,1))
        h = h + h_t.permute(0,2,1)
        h = self.act(self.norm2(self.conv2(h)))
        return h

# -----------------------------------------------------------------------------
# UpBlock: ConvTranspose1d upsampling + FiLM
# -----------------------------------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ctx_dim):
        super().__init__()
        self.up    = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv1d(out_ch * 2, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act   = nn.SiLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.film  = nn.Linear(ctx_dim, out_ch * 2)

    def forward(self, x, skip, ctx):
        # x: [B, in_ch, T_down]; skip: [B, out_ch, T_down]
        h_var = self.up(x)                            # [B, out_ch, T]
        h_var = torch.cat([h_var, skip], dim=1)           # concat skip
        h_var = self.act(self.norm1(self.conv1(h_var)))
        scale, shift = self.film(ctx).chunk(2, dim=-1)

        # Fix broadcasting
        scale = scale.view(scale.size(0), scale.size(1), 1)
        shift = shift.view(shift.size(0), shift.size(1), 1)

        h_var = h_var * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h_var = self.act(self.norm2(self.conv2(h_var)))
        return h_var

# -----------------------------------------------------------------------------
# TrajectoryUNet: full U-Net over time, conditioned via FiLM
# -----------------------------------------------------------------------------
class TrajectoryUNet(nn.Module):
    def __init__(self, max_agents, T, state_dim, feat_dim, hid_dim, ctx_dim):
        """
        max_agents : maximum number of agents (padding size)
        T          : time horizon
        feat_dim   : hidden dims from AxialPreprocessor
        hid_dim    : internal U-Net channel size
        ctx_dim    : conditioning vector size
        """
        super().__init__()
        self.max_agents = max_agents
        self.T = T
        # input channels = max_agents * feat_dim
        in_ch = max_agents * feat_dim
        # output channels = max_agents * 2 (x,y)
        out_ch = max_agents * state_dim

        self.init_conv = nn.Conv1d(in_ch, hid_dim, kernel_size=3, padding=1)
        self.down1 = DownBlock(hid_dim, hid_dim * 2, ctx_dim)
        self.down2 = DownBlock(hid_dim * 2, hid_dim * 4, ctx_dim)
        self.mid   = MiddleBlock(hid_dim * 4, ctx_dim)
        self.up1   = UpBlock(hid_dim * 4, hid_dim * 2, ctx_dim)
        self.up2   = UpBlock(hid_dim * 2, hid_dim,     ctx_dim)
        self.final = nn.Conv1d(hid_dim, out_ch, kernel_size=1)

    def forward(self, feat, ctx):
        """
        feat: [B, Na, T, feat_dim]
        ctx : [B, ctx_dim]
        returns: [B, Na, T, 2]
        """
        B, Na, T, D = feat.shape
        x = feat.permute(0,1,3,2).reshape(B, Na * D, T)
        x_conv = self.init_conv(x)                     # [B, hid, T]

        d1 = self.down1(x_conv, ctx)                   # [B, 2*hid, T/2]
        d2 = self.down2(d1, ctx)                       # [B, 4*hid, T/4]
        m  = self.mid(d2, ctx)                         # [B, 4*hid, T/4]
        u1 = self.up1(m, d1, ctx)                      # [B, 2*hid, T/2] 
        u2 = self.up2(u1, x_conv, ctx)                 # [B, hid, T] 
        out = self.final(u2)                           # [B, out_ch, T]

        return out.view(B, Na, 2, T).permute(0, 1, 3, 2)


