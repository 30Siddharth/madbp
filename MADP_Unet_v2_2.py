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
            alphas_cum = torch.clamp(alphas_cum, min=0.7)
        
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
        
        # Encode agent count - FIXED: handle both tensor and int
        if isinstance(n_agents, torch.Tensor):
            # Take max for batch processing
            n_agents_emb = self.count_emb(n_agents.max())
            n_agents_emb = n_agents_emb.unsqueeze(0).expand(B, -1)
        else:
            n_agents_emb = self.count_emb(torch.tensor(n_agents, device=f_img.device))
            n_agents_emb = n_agents_emb.unsqueeze(0).expand(B, -1)
        
        # Multi-modal fusion
        tokens = torch.stack([f_img, f_pose, n_agents_emb], dim=1)
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
        
        # Add positional embeddings (handle variable T)
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
# FIXED U-Net Building Blocks with Correct Channel Calculations
# -----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ctx_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # FiLM conditioning
        self.film = nn.Linear(ctx_dim, out_ch * 2)
        
        # FIXED: Always use projection for skip connection to handle channel mismatch
        self.skip = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x, ctx):
        skip = self.skip(x)
        
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Apply FiLM conditioning with proper broadcasting
        film_params = self.film(ctx)  # [B, out_ch * 2]
        scale, shift = film_params.chunk(2, dim=-1)  # [B, out_ch] each
        
        # Proper broadcasting: [B, out_ch] -> [B, out_ch, 1]
        scale = scale.unsqueeze(-1)  # [B, out_ch, 1]
        shift = shift.unsqueeze(-1)  # [B, out_ch, 1]
        
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.dropout(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        
        return self.act(h + skip)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ctx_dim):
        super().__init__()
        self.res_block = ResidualBlock(in_ch, out_ch, ctx_dim)
        self.downsample = nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, ctx):
        h = self.res_block(x, ctx)
        return self.downsample(h), h  # Return downsampled and skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, ctx_dim):
        super().__init__()
        # Proper upsampling that handles size mismatches
        self.upsample = nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1)
        
        # FIXED: After concatenation, input channels = out_ch + skip_ch
        self.res_block = ResidualBlock(out_ch + skip_ch, out_ch, ctx_dim)

    def forward(self, x, skip, ctx):
        # Upsample
        h = self.upsample(x)
        
        # Handle size mismatches between upsampled and skip
        if h.size(-1) != skip.size(-1):
            # Interpolate to match skip connection size
            h = F.interpolate(h, size=skip.size(-1), mode='linear', align_corners=False)
        
        # Concatenate with skip connection
        h = torch.cat([h, skip], dim=1)
        
        return self.res_block(h, ctx)

class MiddleBlock(nn.Module):
    def __init__(self, ch, ctx_dim):
        super().__init__()
        self.res_block1 = ResidualBlock(ch, ch, ctx_dim)
        self.attn = nn.MultiheadAttention(ch, 8, batch_first=True)
        self.res_block2 = ResidualBlock(ch, ch, ctx_dim)

    def forward(self, x, ctx):
        h = self.res_block1(x, ctx)
        
        # Self-attention
        h_attn = h.permute(0, 2, 1)  # [B, T, ch]
        h_attn, _ = self.attn(h_attn, h_attn, h_attn)
        h = h + h_attn.permute(0, 2, 1)  # Add residual
        
        h = self.res_block2(h, ctx)
        return h

# -----------------------------------------------------------------------------
# FIXED Enhanced Trajectory U-Net with Correct Channel Calculations
# -----------------------------------------------------------------------------

class EnhancedTrajectoryUNet(nn.Module):
    def __init__(self, max_agents, T, state_dim, feat_dim, hid_dim, ctx_dim):
        super().__init__()
        self.max_agents = max_agents
        self.T = T
        
        # Input channels should be feat_dim (from preprocessor output)
        in_ch = feat_dim
        out_ch = state_dim  # Output 2D coordinates per agent
        
        # Initial convolution
        self.init_conv = nn.Conv1d(in_ch, hid_dim, 3, padding=1)
        
        # Encoder with proper channel progression
        self.down1 = DownBlock(hid_dim, hid_dim * 2, ctx_dim)
        self.down2 = DownBlock(hid_dim * 2, hid_dim * 4, ctx_dim)
        self.down3 = DownBlock(hid_dim * 4, hid_dim * 8, ctx_dim)
        
        # Middle
        self.middle = MiddleBlock(hid_dim * 8, ctx_dim)
        
        # FIXED Decoder with correct skip connection channel calculations
        # up1: input=8*hid, skip=8*hid, output=4*hid
        self.up1 = UpBlock(hid_dim * 8, hid_dim * 8, hid_dim * 4, ctx_dim)
        
        # up2: input=4*hid, skip=4*hid, output=2*hid  
        self.up2 = UpBlock(hid_dim * 4, hid_dim * 4, hid_dim * 2, ctx_dim)
        
        # up3: input=2*hid, skip=2*hid, output=hid
        self.up3 = UpBlock(hid_dim * 2, hid_dim * 2, hid_dim, ctx_dim)
        
        # Output
        self.final_conv = nn.Sequential(
            nn.Conv1d(hid_dim, hid_dim, 3, padding=1),
            nn.GroupNorm(min(8, hid_dim), hid_dim),
            nn.SiLU(),
            nn.Conv1d(hid_dim, out_ch, 1)
        )

    def forward(self, feat, ctx):
        B, Na, T, D = feat.shape
        
        # Process each agent separately, then combine
        # Reshape to process all agents in batch: [B*Na, D, T]
        x = feat.view(B * Na, D, T)
        
        # Expand context for all agents: [B*Na, ctx_dim]
        ctx_expanded = ctx.unsqueeze(1).expand(B, Na, -1).reshape(B * Na, -1)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder
        x1, skip1 = self.down1(x, ctx_expanded)      # skip1: [B*Na, 2*hid, T/2]
        x2, skip2 = self.down2(x1, ctx_expanded)     # skip2: [B*Na, 4*hid, T/4]
        x3, skip3 = self.down3(x2, ctx_expanded)     # skip3: [B*Na, 8*hid, T/8]
        
        # Middle
        x = self.middle(x3, ctx_expanded)            # x: [B*Na, 8*hid, T/8]
        
        # FIXED Decoder with correct skip connection order and channels
        x = self.up1(x, skip3, ctx_expanded)         # x: [B*Na, 4*hid, T/4]
        x = self.up2(x, skip2, ctx_expanded)         # x: [B*Na, 2*hid, T/2]
        x = self.up3(x, skip1, ctx_expanded)         # x: [B*Na, hid, T]
        
        # Output
        out = self.final_conv(x)  # [B*Na, 2, T]
        
        # Reshape back to [B, Na, T, 2]
        out = out.view(B, Na, 2, T).permute(0, 1, 3, 2)
        
        return out
