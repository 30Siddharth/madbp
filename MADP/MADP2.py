import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class MultiAgentDiffusionPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int = 6,  # Dimension of state-action per agent
        context_dim: int = 64,  # Dimension of context tokens
        max_num_agents: int = 10,
        horizon: int = 10,
        hidden_dim: int = 256,
        n_diffusion_steps: int = 1000,
        beta_schedule: str = "linear"
    ):
        super().__init__()
        self.state_dim = state_dim
        self.max_num_agents = max_num_agents
        self.horizon = horizon
        self.n_diffusion_steps = n_diffusion_steps
        
        # Define betas for diffusion process (noise scheduler)
        if beta_schedule == "linear":
            self.betas = torch.linspace(1e-4, 2e-2, n_diffusion_steps)
        elif beta_schedule == "cosine":
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = n_diffusion_steps + 1
            x = torch.linspace(0, n_diffusion_steps, steps)
            alphas_cumprod = torch.cos(((x / n_diffusion_steps) + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        
        # Pre-compute diffusion parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # U-Net for trajectory denoising
        self.trajectory_unet = TrajectoryUNet(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        )
    
    def get_time_embedding(self, timesteps, hidden_dim):
        """
        Create sinusoidal time embeddings.
        """
        half_dim = hidden_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.time_embed(emb)
    
    def forward_diffusion(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def p_losses(self, x_start, context, t, noise=None, loss_type="l2"):
        """
        Training loss computation
        """
        B, T, n_agents, D = x_start.shape  # Batch, Time, Agents, Dimension
        
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Forward diffusion (add noise)
        x_noisy, noise = self.forward_diffusion(x_start=x_start, t=t, noise=noise)
        
        # Get context features
        context_features = self.context_encoder(context)
        
        # Get time embedding
        time_emb = self.get_time_embedding(t, hidden_dim=context_features.shape[-1])
        
        # Predict noise with U-Net (denoising step)
        predicted_noise = self.trajectory_unet(x_noisy, time_emb, context_features, n_agents)
        
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
            
        return loss
    
    def forward(self, x, context):
        """
        Forward pass during training
        """
        b, *_ = x.shape
        t = torch.randint(0, self.n_diffusion_steps, (b,), device=x.device).long()
        return self.p_losses(x, context, t)
    
    @torch.no_grad()
    def p_sample(self, x, context_features, t, t_index, n_agents):
        """
        Sample from the model at timestep t (denoising step)
        """
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        
        # Get time embedding
        time_emb = self.get_time_embedding(t, hidden_dim=context_features.shape[-1])
        
        # Predict noise with U-Net
        predicted_noise = self.trajectory_unet(x, time_emb, context_features, n_agents)
        
        # Equation 11 in the DDPM paper
        # Use our model (noise predictor) to predict the mean
        mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample_trajectory(self, context, n_agents, shape=None):
        """
        Sample a trajectory from the model (inference)
        """
        if shape is None:
            batch_size = context.shape[0]
            shape = (batch_size, self.horizon, n_agents, self.state_dim)
            
        device = next(self.parameters()).device
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Get context features once
        context_features = self.context_encoder(context)
        
        # Iterative denoising process
        for i in reversed(range(0, self.n_diffusion_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, context_features, t, i, n_agents)
            
        return x


class TrajectoryUNet(nn.Module):
    """
    U-Net architecture for trajectory denoising with attention mechanisms
    """
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.proj = nn.Linear(state_dim, hidden_dim)
        
        # Down blocks
        self.down1 = DownBlock(hidden_dim, hidden_dim * 2)
        self.down2 = DownBlock(hidden_dim * 2, hidden_dim * 4)
        
        # Middle block with attention for agent interaction
        self.mid = MiddleBlock(hidden_dim * 4, hidden_dim * 4)
        
        # Up blocks
        self.up1 = UpBlock(hidden_dim * 4, hidden_dim * 2)
        self.up2 = UpBlock(hidden_dim * 2, hidden_dim)
        
        # Cross-attention for agent interaction (between down and up sampling)
        self.cross_attention1 = CrossAttention(hidden_dim * 4)
        self.cross_attention2 = CrossAttention(hidden_dim * 2)
        
        # Final layers
        self.final = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, state_dim, 1)
        )
        
        # Condition projection
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        
    def forward(self, x, time_emb, context_features, n_agents):
        """
        Forward pass of the U-Net
        
        Args:
            x: Tensor of shape (batch_size, horizon, n_agents, state_dim)
            time_emb: Time embedding of shape (batch_size, hidden_dim)
            context_features: Context features of shape (batch_size, hidden_dim)
            n_agents: Number of agents to process
        """
        batch_size, horizon, _, _ = x.shape
        
        # Reshape for 1D convolutions: (batch_size * n_agents, state_dim, horizon)
        x = x[:, :, :n_agents, :].reshape(batch_size * n_agents, horizon, self.state_dim)
        x = x.permute(0, 2, 1)  # -> (batch_size * n_agents, state_dim, horizon)
        
        # Initial projection
        x = self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)  # -> (batch_size * n_agents, hidden_dim, horizon)
        
        # Combine time and context conditions
        cond = time_emb + context_features
        cond = self.cond_proj(cond)
        
        # Repeat condition for each agent
        cond = cond.repeat_interleave(n_agents, dim=0)
        
        # U-Net forward pass with skip connections
        h1 = self.down1(x, cond)
        h2 = self.down2(h1, cond)
        
        # Middle block
        h = self.mid(h2, cond)
        
        # Cross-attention for agent interaction (between down and up sampling)
        h = self.cross_attention1(h, batch_size, n_agents)
        
        # Up blocks with skip connections
        h = self.up1(h, h2, cond)
        
        # Another cross-attention layer
        h = self.cross_attention2(h, batch_size, n_agents)
        
        # Final up block
        h = self.up2(h, h1, cond)
        
        # Final layers
        output = self.final(h)
        
        # Reshape back to original format
        output = output.permute(0, 2, 1)  # -> (batch_size * n_agents, horizon, state_dim)
        output = output.reshape(batch_size, n_agents, horizon, self.state_dim)
        output = output.permute(0, 2, 1, 3)  # -> (batch_size, horizon, n_agents, state_dim)
        
        # Pad with zeros for agents beyond n_agents if needed
        if n_agents < self.max_num_agents:
            padding = torch.zeros(batch_size, horizon, self.max_num_agents - n_agents, self.state_dim, 
                                 device=output.device)
            output = torch.cat([output, padding], dim=2)
            
        return output


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.downsample = nn.AvgPool1d(2)
        
        # Condition projection
        self.cond_proj = nn.Linear(out_channels, out_channels * 2)
        
    def forward(self, x, cond):
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Apply condition (FiLM conditioning)
        cond_proj = self.cond_proj(cond).unsqueeze(-1)
        scale, shift = torch.chunk(cond_proj, 2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.act(h)
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        # Downsample
        return self.downsample(h)


class MiddleBlock(nn.Module):
    """
    Middle block with self-attention for agent interaction
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # Self-attention for sequence modeling
        self.attention = nn.MultiheadAttention(out_channels, 4, batch_first=True)
        
        # Condition projection
        self.cond_proj = nn.Linear(out_channels, out_channels * 2)
        
    def forward(self, x, cond):
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Apply condition (FiLM conditioning)
        cond_proj = self.cond_proj(cond).unsqueeze(-1)
        scale, shift = torch.chunk(cond_proj, 2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.act(h)
        
        # Self-attention
        h_attn = h.permute(0, 2, 1)  # -> (batch_size * n_agents, seq_len, channels)
        h_attn, _ = self.attention(h_attn, h_attn, h_attn)
        h_attn = h_attn.permute(0, 2, 1)  # -> (batch_size * n_agents, channels, seq_len)
        
        # Residual connection
        h = h + h_attn
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h


class UpBlock(nn.Module):
    """
    Upsampling block for U-Net
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels * 2, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        
        # Condition projection
        self.cond_proj = nn.Linear(out_channels, out_channels * 2)
        
    def forward(self, x, skip, cond):
        # Upsample
        h = self.upsample(x)
        
        # Concatenate with skip connection
        h = torch.cat([h, skip], dim=1)
        
        # First conv block
        h = self.conv1(h)
        h = self.norm1(h)
        
        # Apply condition (FiLM conditioning)
        cond_proj = self.cond_proj(cond).unsqueeze(-1)
        scale, shift = torch.chunk(cond_proj, 2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.act(h)
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h


class CrossAttention(nn.Module):
    """
    Cross-attention for agent interaction
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, batch_size, n_agents):
        """
        Forward pass for cross-attention

        Args:
            x: Tensor of shape (batch_size * n_agents, hidden_dim, seq_len)
            batch_size: Number of batches
            n_agents: Number of agents

        Returns:
            Tensor of shape (batch_size * n_agents, hidden_dim, seq_len)
        """
        # Reshape for attention: (batch_size, n_agents, seq_len, hidden_dim)
        x = x.view(batch_size, n_agents, x.size(2), self.hidden_dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * x.size(2), n_agents, self.hidden_dim)

        # Apply multihead attention
        x_attn, _ = self.attention(x, x, x)

        # Reshape back to original format
        x_attn = x_attn.view(batch_size, x.size(1), n_agents, self.hidden_dim)
        x_attn = x_attn.permute(0, 2, 1, 3).reshape(batch_size * n_agents, self.hidden_dim, x.size(1))

        # Apply layer normalization
        x_attn = self.norm(x_attn.permute(0, 2, 1)).permute(0, 2, 1)

        return x_attn
