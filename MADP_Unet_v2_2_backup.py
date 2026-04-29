import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pdb

# -----------------------------------------------------------------------------
# Enhanced Noise Scheduler with Boundary-Aware Sampling
# -----------------------------------------------------------------------------

class BoundaryAwareNoiseScheduler(nn.Module):
    def __init__(self, T=1000, schedule_type='cosine', max_noise_ratio=2.5):
        super().__init__()
        self.T = T
        self.schedule_type = schedule_type
        
        if schedule_type == 'cosine':
            # Cosine schedule with controlled noise levels
            steps = torch.arange(T, dtype=torch.float32)
            s = 0.008
            alphas_cum = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cum = alphas_cum / alphas_cum[0]
            
            # Prevent excessive noise that could violate boundary constraints
            min_alpha = 1.0 / (1.0 + max_noise_ratio**2)
            alphas_cum = torch.clamp(alphas_cum, min=min_alpha, max=1.0)
            
        elif schedule_type == 'linear':
            # Linear schedule with boundary preservation
            betas = torch.linspace(1e-5, 2e-3, T)
            alphas = 1 - betas
            alphas_cum = torch.cumprod(alphas, dim=0)
            # Ensure minimum signal retention for boundary constraints
            alphas_cum = torch.clamp(alphas_cum, min=0.1)
        
        # Compute derived quantities
        betas = torch.zeros(T)
        betas[0] = 1 - alphas_cum[0]
        betas[1:] = 1 - (alphas_cum[1:] / alphas_cum[:-1])
        betas = torch.clamp(betas, min=1e-8, max=0.999)
        
        alphas = 1 - betas
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cum', alphas_cum)
        self.register_buffer('sqrt_ac', torch.sqrt(alphas_cum))
        self.register_buffer('sqrt_1mac', torch.sqrt(1 - alphas_cum))
    
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion with boundary preservation"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Apply noise scheduling
        if len(x0.shape) == 4:  # [B, Na, 2, T]
            a = self.sqrt_ac[t].view(-1, 1, 1, 1)
            am = self.sqrt_1mac[t].view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unexpected tensor shape: {x0.shape}")
        
        noisy_x = a * x0 + am * noise
        return noisy_x, noise

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * (torch.arange(half, device=t.device) / (half - 1)))
        args = t.unsqueeze(1).float() * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

# -----------------------------------------------------------------------------
# Context Encoder with Multi-Modal Fusion
# -----------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    def __init__(self, img_ch=3, pose_dim=2, hid=128, max_agents=10):
        super().__init__()
        self.max_agents = max_agents
        
        # Image encoder
        self.img_enc = nn.Sequential(
            nn.Conv2d(img_ch, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, hid)
        )
        
        # Pose encoder for start/goal positions
        self.pose_enc = nn.Sequential(
            nn.Linear(2 * max_agents * 2, hid * 2),
            nn.ReLU(),
            nn.Linear(hid * 2, hid),
            nn.ReLU(),
            nn.Linear(hid, hid)
        )
        
        # Agent count embedding
        self.count_emb = nn.Embedding(max_agents + 1, hid)
        
        # Multi-modal fusion
        self.fusion = nn.MultiheadAttention(hid, 8, batch_first=True)
        self.norm = nn.LayerNorm(hid)
        
    def forward(self, img, start, goal, n_agents):
        B = img.size(0)
        
        # Encode image features
        f_img = self.img_enc(img)
        
        # Encode pose information
        poses = torch.cat([start, goal], dim=-1).flatten(1)
        f_pose = self.pose_enc(poses)
        
        # Encode agent count
        f_cnt = self.count_emb(n_agents)
        
        # Multi-modal fusion
        tokens = torch.stack([f_img, f_pose, f_cnt], dim=1)
        fused, _ = self.fusion(tokens, tokens, tokens)
        
        return self.norm(fused.mean(1))

# -----------------------------------------------------------------------------
# Axial Attention Preprocessor
# -----------------------------------------------------------------------------

class AxialPreprocessor(nn.Module):
    def __init__(self, state_dim=2, feat_dim=128, num_heads=8, max_agents=10, T=20):
        super().__init__()
        self.max_agents = max_agents
        self.T = T
        
        # Input projection
        self.input_proj = nn.Linear(state_dim, feat_dim)
        
        # Positional encodings
        self.agent_pos_emb = nn.Parameter(torch.randn(max_agents, feat_dim))
        self.time_pos_emb = nn.Parameter(torch.randn(T, feat_dim))
        
        # Axial attention layers
        self.agent_attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.time_attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim * 2, feat_dim)
        )
        
    def forward(self, x, agent_mask):
        B, Na, T, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional embeddings
        x = x + self.agent_pos_emb[:Na].unsqueeze(0).unsqueeze(2)
        x = x + self.time_pos_emb[:T].unsqueeze(0).unsqueeze(1)
        
        # Agent-axis attention
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, Na, -1)
        mask_flat = agent_mask.unsqueeze(1).expand(B, T, Na).reshape(B * T, Na)
        
        x_agent, _ = self.agent_attn(x_flat, x_flat, x_flat, key_padding_mask=mask_flat)
        x_agent = x_agent.view(B, T, Na, -1).permute(0, 2, 1, 3)
        x = self.norm1(x + x_agent)
        
        # Time-axis attention
        x_flat = x.reshape(B * Na, T, -1)
        x_time, _ = self.time_attn(x_flat, x_flat, x_flat)
        x_time = x_time.view(B, Na, T, -1)
        x = self.norm2(x + x_time)
        
        # MLP
        x = x + self.mlp(x)
        
        return x

# -----------------------------------------------------------------------------
# U-Net Building Blocks
# -----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ctx_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # FiLM conditioning
        self.film = nn.Linear(ctx_dim, out_ch * 2)
        
        # Skip connection
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, ctx):
        skip = self.skip(x)
        
        var = self.conv1(x)
        var = self.norm1(var)
        
        # Apply FiLM conditioning
        film_params = self.film(ctx)
        scale, shift = film_params.chunk(2, dim=-1)
        scale = scale.view(scale.size(0), scale.size(1), 1)
        shift = shift.view(shift.size(0), shift.size(1), 1)
        var = var * (1 + scale) + shift
        
        var = self.act(var)
        var = self.dropout(var)
        var = self.conv2(var)
        var = self.norm2(var)
        
        return self.act(var + skip)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ctx_dim):
        super().__init__()
        self.res_block = ResidualBlock(in_ch, out_ch, ctx_dim)
        self.downsample = nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1)
        
    def forward(self, x, ctx):
        var = self.res_block(x, ctx)
        return self.downsample(var), var

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ctx_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res_block = ResidualBlock(out_ch * 2, out_ch, ctx_dim)
        
    def forward(self, x, skip, ctx):
        var = self.upsample(x)
        var = torch.cat([var, skip], dim=1)
        return self.res_block(var, ctx)

class MiddleBlock(nn.Module):
    def __init__(self, ch, ctx_dim):
        super().__init__()
        self.res_block1 = ResidualBlock(ch, ch, ctx_dim)
        self.attn = nn.MultiheadAttention(ch, 8, batch_first=True)
        self.res_block2 = ResidualBlock(ch, ch, ctx_dim)
        
    def forward(self, x, ctx):
        var = self.res_block1(x, ctx)
        
        # Self-attention
        h_attn = var.permute(0, 2, 1)
        h_attn, _ = self.attn(h_attn, h_attn, h_attn)
        var = var + h_attn.permute(0, 2, 1)
        
        var = self.res_block2(var, ctx)
        return var

# -----------------------------------------------------------------------------
# Enhanced Trajectory U-Net
# -----------------------------------------------------------------------------

class EnhancedTrajectoryUNet(nn.Module):
    def __init__(self, max_agents, T, state_dim, feat_dim, hid_dim, ctx_dim):
        super().__init__()
        self.max_agents = max_agents
        self.T = T
        
        # Input/output dimensions
        in_ch = max_agents * feat_dim
        out_ch = max_agents * state_dim
        
        # Initial convolution
        self.init_conv = nn.Conv1d(in_ch, hid_dim, 3, padding=1)
        
        # Encoder
        self.down1 = DownBlock(hid_dim, hid_dim * 2, ctx_dim)
        self.down2 = DownBlock(hid_dim * 2, hid_dim * 4, ctx_dim)
        self.down3 = DownBlock(hid_dim * 4, hid_dim * 8, ctx_dim)
        
        # Middle
        self.middle = MiddleBlock(hid_dim * 8, ctx_dim)
        
        # Decoder
        self.up1 = UpBlock(hid_dim * 8, hid_dim * 4, ctx_dim)
        self.up2 = UpBlock(hid_dim * 4, hid_dim * 2, ctx_dim)
        self.up3 = UpBlock(hid_dim * 2, hid_dim, ctx_dim)
        
        # Output
        self.final_conv = nn.Sequential(
            nn.Conv1d(hid_dim, hid_dim, 3, padding=1),
            nn.GroupNorm(8, hid_dim),
            nn.SiLU(),
            nn.Conv1d(hid_dim, out_ch, 1)
        )
        
    def forward(self, feat, ctx):
        B, Na, T, D = feat.shape
        
        # Reshape for 1D convolution: [B, Na*D, T]
        x = feat.permute(0, 1, 3, 2).reshape(B, Na * D, T)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder
        x1, skip1 = self.down1(x, ctx)
        x2, skip2 = self.down2(x1, ctx)
        x3, skip3 = self.down3(x2, ctx)
        
        # Middle
        x = self.middle(x3, ctx)
        
        # Decoder
        pdb.set_trace()
        x = self.up1(x, skip3, ctx)
        x = self.up2(x, skip2, ctx)
        x = self.up3(x, skip1, ctx)
        
        # Output
        out = self.final_conv(x)
        
        # Reshape back: [B, Na, T, 2]
        return out.view(B, Na, 2, T).permute(0, 1, 3, 2)
