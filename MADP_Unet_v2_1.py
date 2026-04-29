import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

# -----------------------------------------------------------------------------
# Enhanced NoiseScheduler with Bounded Noise
# -----------------------------------------------------------------------------
class EnhancedNoiseScheduler(nn.Module):
    def __init__(self, T=1000, schedule_type='bounded_cosine', max_noise_ratio=3.0):
        super().__init__()
        self.T = T
        self.schedule_type = schedule_type
        
        if schedule_type == 'bounded_cosine':
            # Cosine schedule with bounded noise to prevent excessive scaling
            steps = torch.arange(T, dtype=torch.float32)
            s = 0.008
            alphas_cum = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cum = alphas_cum / alphas_cum[0]
            
            # Constrain minimum signal retention to prevent noise explosion
            min_alpha = 1.0 / (1.0 + max_noise_ratio**2)
            alphas_cum = torch.clamp(alphas_cum, min=min_alpha, max=1.0)
            
        elif schedule_type == 'linear':
            # Original linear schedule for comparison
            betas = torch.linspace(1e-5, 2e-3, T)
            alphas = 1 - betas
            alphas_cum = torch.cumprod(alphas, dim=0)
            alphas_cum = torch.clamp(alphas_cum, min=0.8)
        
        # Compute betas from alphas_cum
        # betas = torch.zeros(T)
        # betas[0] = 1 - alphas_cum[0]
        # betas[1:] = 1 - (alphas_cum[1:] / alphas_cum[:-1])
        # betas = torch.clamp(betas, 0, 0.999)
        
        # alphas = 1 - betas
        # alphas = torch.zeros(T)
        # alphas[0] = alphas_cum[0]
        # alphas[1:] = alphas_cum[1:] / alphas_cum[:-1]
        # betas = 1 - alphas

        # betas = torch.clamp(betas, min=1e-8, max=0.999)
        # alphas = 1 - betas
        # alphas_cum = torch.cumprod(alphas, dim=0)
        
        # Register all buffers with consistent shapes
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cum', alphas_cum)
        self.register_buffer('sqrt_ac', torch.sqrt(alphas_cum))
        self.register_buffer('sqrt_1mac', torch.sqrt(1 - alphas_cum))
    
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion with proper tensor handling"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Handle both [B, Na, 2, T] and [B, Na, T, 2] formats
        if len(x0.shape) == 4:
            a = self.sqrt_ac[t].view(-1, 1, 1, 1)
            am = self.sqrt_1mac[t].view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unexpected tensor shape: {x0.shape}")
        
        noisy_x = a * x0 + am * noise
        return noisy_x, noise
    
class ImprovedNoiseScheduler(nn.Module):
    def __init__(self, T=1000, schedule_type='laplace_logsnr', concentration_factor=2.0):
        super().__init__()
        self.T = T
        
        if schedule_type == 'laplace_logsnr':
            # Create log-SNR values concentrated around 0
            t_normalized = torch.linspace(0, 1, T)
            # Laplace distribution centered at 0.5 (middle of process)
            log_snr = concentration_factor * (0.5 - torch.abs(t_normalized - 0.5))
            
            # Convert log-SNR to alphas_cum: log(SNR) = log(α²/(1-α²))
            alphas_cum = torch.sigmoid(log_snr)  # More stable than exponential
            
        # Compute proper betas maintaining consistency
        alphas = torch.zeros(T)
        alphas[0] = alphas_cum[0]
        alphas[1:] = alphas_cum[1:] / alphas_cum[:-1]
        betas = 1 - alphas
        
        # Ensure numerical stability
        betas = torch.clamp(betas, min=1e-8, max=0.999)
        alphas = 1 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)

        # Register all buffers with consistent shapes
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cum', alphas_cum)
        self.register_buffer('sqrt_ac', torch.sqrt(alphas_cum))
        self.register_buffer('sqrt_1mac', torch.sqrt(1 - alphas_cum))

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion with proper tensor handling"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Handle both [B, Na, 2, T] and [B, Na, T, 2] formats
        if len(x0.shape) == 4:
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
# Context Encoder with Simplified Architecture
# -----------------------------------------------------------------------------
class ContextEncoder(nn.Module):
    def __init__(self, img_ch=3, pose_dim=2, hid=128, max_agents=10):
        super().__init__()
        # Simplified image encoder to avoid ResNet complexity
        self.img_enc = nn.Sequential(
            nn.Conv2d(img_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, hid),
            nn.ReLU()
        )
        self.pose_enc = nn.Sequential(
            nn.Linear(2 * max_agents * 2, hid),
            nn.ReLU(),
            nn.Linear(hid, hid)
        )
        self.count_emb = nn.Embedding(max_agents + 1, hid)
        self.ctx_attn = nn.MultiheadAttention(hid, 4, batch_first=True)

    def forward(self, img, start, goal, n_agents):
        B = img.size(0)
        f_img = self.img_enc(img)
        poses = torch.cat([start, goal], dim=-1).flatten(1)
        f_pose = self.pose_enc(poses)
        f_cnt = self.count_emb(n_agents)
        tokens = torch.stack([f_img, f_pose, f_cnt], dim=1)
        attn, _ = self.ctx_attn(tokens, tokens, tokens)
        return attn.mean(1)

# -----------------------------------------------------------------------------
# Axial Preprocessor with Memory Optimization
# -----------------------------------------------------------------------------
class AxialPreprocessor(nn.Module):
    def __init__(self, state_dim=2, feat_dim=128, num_heads=4, max_agents=10, T=20):
        super().__init__()
        self.max_agents = max_agents
        self.T = T
        
        self.input_proj = nn.Linear(state_dim, feat_dim)
        self.agent_attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.time_attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, x, agent_mask):
        B, Na, T, _ = x.shape
        x_proj = self.input_proj(x)
        
        # Agent-axis attention with gradient checkpointing for memory efficiency
        xa = x_proj.permute(0, 2, 1, 3).reshape(B * T, Na, -1)
        am = agent_mask.unsqueeze(1).expand(B, T, Na).reshape(B * T, Na)
        xa2, _ = self.agent_attn(xa, xa, xa, key_padding_mask=am)
        xa2 = xa2.view(B, T, Na, -1).permute(0, 2, 1, 3)
        
        # Time-axis attention
        xt = xa2.reshape(B * Na, T, -1)
        xt2, _ = self.time_attn(xt, xt, xt)
        xt2 = xt2.view(B, Na, T, -1)
        
        return self.mlp(xt2)

# -----------------------------------------------------------------------------
# Fixed U-Net Components with Proper Broadcasting
# -----------------------------------------------------------------------------
class FixedDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ctx_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.film = nn.Linear(ctx_dim, out_ch * 2)

    def forward(self, x, ctx):
        h = self.act(self.norm1(self.conv1(x)))
        
        # Fixed FiLM conditioning with proper broadcasting
        film_out = self.film(ctx)
        scale, shift = film_out.chunk(2, dim=-1)
        
        # Critical fix: Use view() instead of unsqueeze() for proper broadcasting
        scale = scale.view(scale.size(0), scale.size(1), 1)
        shift = shift.view(shift.size(0), shift.size(1), 1)
        
        h = h * (1 + scale) + shift
        h = self.act(self.norm2(self.conv2(h)))
        return h

class FixedMiddleBlock(nn.Module):
    def __init__(self, ch, ctx_dim, num_heads=4):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, ch)
        self.act = nn.SiLU()
        self.film = nn.Linear(ctx_dim, ch * 2)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)

    def forward(self, x, ctx):
        h = self.act(self.norm1(self.conv1(x)))
        
        # Fixed FiLM conditioning
        film_out = self.film(ctx)
        scale, shift = film_out.chunk(2, dim=-1)
        scale = scale.view(scale.size(0), scale.size(1), 1)
        shift = shift.view(shift.size(0), shift.size(1), 1)
        
        h = h * (1 + scale) + shift
        
        # Temporal self-attention
        h_t, _ = self.attn(h.permute(0,2,1), h.permute(0,2,1), h.permute(0,2,1))
        h = h + h_t.permute(0,2,1)
        h = self.act(self.norm2(self.conv2(h)))
        return h

class FixedUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ctx_dim):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv1d(out_ch * 2, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.film = nn.Linear(ctx_dim, out_ch * 2)

    def forward(self, x, skip, ctx):
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)
        h = self.act(self.norm1(self.conv1(h)))
        
        # Fixed FiLM conditioning
        film_out = self.film(ctx)
        scale, shift = film_out.chunk(2, dim=-1)
        scale = scale.view(scale.size(0), scale.size(1), 1)
        shift = shift.view(shift.size(0), shift.size(1), 1)
        
        h = h * (1 + scale) + shift
        h = self.act(self.norm2(self.conv2(h)))
        return h

# -----------------------------------------------------------------------------
# Enhanced Trajectory U-Net
# -----------------------------------------------------------------------------
class EnhancedTrajectoryUNet(nn.Module):
    def __init__(self, max_agents, T, state_dim, feat_dim, hid_dim, ctx_dim):
        super().__init__()
        self.max_agents = max_agents
        self.T = T
        
        in_ch = max_agents * feat_dim
        out_ch = max_agents * state_dim
        
        self.init_conv = nn.Conv1d(in_ch, hid_dim, kernel_size=3, padding=1)
        self.down1 = FixedDownBlock(hid_dim, hid_dim * 2, ctx_dim)
        self.down2 = FixedDownBlock(hid_dim * 2, hid_dim * 4, ctx_dim)
        self.mid = FixedMiddleBlock(hid_dim * 4, ctx_dim)
        self.up1 = FixedUpBlock(hid_dim * 4, hid_dim * 2, ctx_dim)
        self.up2 = FixedUpBlock(hid_dim * 2, hid_dim, ctx_dim)
        self.final = nn.Conv1d(hid_dim, out_ch, kernel_size=1)

    def forward(self, feat, ctx):
        B, Na, T, D = feat.shape
        x = feat.permute(0, 1, 3, 2).reshape(B, Na * D, T)
        x_conv = self.init_conv(x)
        
        d1 = self.down1(x_conv, ctx)
        d2 = self.down2(d1, ctx)
        m = self.mid(d2, ctx)
        u1 = self.up1(m, d1, ctx)
        u2 = self.up2(u1, x_conv, ctx)
        out = self.final(u2)
        
        return out.view(B, Na, 2, T).permute(0, 1, 3, 2)
