'''
MADP4.py - Streamlined Multi-Agent Diffusion Policy
Author: Siddharth Singh
Date: 04.07.2025
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import pdb

import matplotlib.pyplot as plt
import seaborn as sns

device = 'cuda' if torch.cuda.is_available() else "cpu"

class MultiAgentDiffusionPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int = 6,
        image_channels: int = 3,
        context_dim: int = 64,
        max_num_agents: int = 10,
        horizon: int = 10,
        hidden_dim: int = 128,  # Reduced from 256
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
            betas = torch.linspace(1e-4, 2e-2, n_diffusion_steps)
        elif beta_schedule == "cosine":
            # Cosine schedule
            steps = n_diffusion_steps + 1
            x = torch.linspace(0, n_diffusion_steps, steps)
            alphas_cumprod = torch.cos(((x / n_diffusion_steps) + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        
        self.register_buffer("betas", betas)
        
        # Pre-compute diffusion parameters
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        
        # Image Encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim // 4)
        )
        
        # State Encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(4 * max_num_agents, hidden_dim // 2),    # 4 => [x0, y0, x_dot0, y_dot0]
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # Goal State Encoder, Note: In case the goal is the same, this can just be identical to the state-encoder else based for a special case this
        # can be altered to introduce an embedding - or an image based encoder.
        self.goal_encoder = nn.Sequential(
            nn.Linear(2 * max_num_agents, hidden_dim // 2),   # 2 => [x*, y*]
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # Agent encoder: This is encoded just so that it as a similar scale when concatenated for conditioning the Diffusion model.
        self.agent_count_encoder = nn.Embedding(max_num_agents + 1, hidden_dim // 4)
        
        # Add cross-attention between context elements
        self.context_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # U-Net for trajectory denoising
        self.trajectory_unet = TrajectoryUNet(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            context_dim=hidden_dim
        )
    
    def process_context(self, context):
        '''
        Process the heterogenous context tokens.
        '''
        # Unpack context
        # TODO: Update this unpaking for dictionary/hash-map based context.
        images, start_states, goal_states, num_agents = context
        
        # Handle time dimension in images (B, T, C, H, W)
        b, t, c, h, w = images.shape
        images_reshaped = images.reshape(b * t, c, h, w)
        img_features = self.image_encoder(images_reshaped)
        img_features = img_features.reshape(b, t, -1).mean(dim=1)  # Average over time
        
        # Handle start states (B, Na, 4)
        start_features = self.state_encoder(start_states.reshape(b, -1))
        
        # Handle goal states (B, Na, 2)
        goal_features = self.goal_encoder(goal_states.reshape(b, -1))
        
        # Use the number of agents directly
        # agent_features = self.agent_count_encoder(num_agents)
        agent_features = self.agent_count_encoder(num_agents).squeeze(1)

        # Combine features
        combined = torch.cat([img_features, start_features, goal_features, agent_features], dim=1)
        
        # Apply attention for better integration
        context_features = self.context_attention(combined, combined, combined)[0]
        
        features_np = context_features.detach().cpu().numpy()

        plt.figure(figsize=(12, 10))
        sns.heatmap(features_np, cmap='viridis')
        plt.title('Context Features Heatmap')
        plt.savefig('context_features_heatmap.png')
        plt.show()
        
        return context_features
    
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
        context_features = self.process_context(context)
        
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
            batch_size = context[0].shape[0]  # Get batch size from first context element
            shape = (batch_size, self.horizon, n_agents, self.state_dim)
        
        device = next(self.parameters()).device
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Get context features once
        context_features = self.process_context(context)
        
        # Iterative denoising process
        for i in reversed(range(0, self.n_diffusion_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, context_features, t, i, n_agents)
        
        return x

class TrajectoryUNet(nn.Module):
    """
    U-Net architecture for trajectory denoising with attention mechanisms
    """
    def __init__(self, state_dim, hidden_dim, context_dim):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.proj = nn.Linear(state_dim, hidden_dim)
        
        # Down blocks with reduced dimensions
        self.down1 = DownBlock(hidden_dim, hidden_dim*2, context_dim)
        self.down2 = DownBlock(hidden_dim*2, hidden_dim*3, context_dim)  # Reduced from 4x to 3x
        
        # Middle block
        self.mid = MiddleBlock(hidden_dim*3, context_dim)
        
        # Up blocks
        # self.up1 = UpBlock(hidden_dim*3, hidden_dim*2, context_dim)  
        # self.up2 = UpBlock(hidden_dim*2, hidden_dim, context_dim)

        # Up blocks with correct input dimensions after concatenation
        self.up1 = UpBlock(hidden_dim*3*2, hidden_dim*2, context_dim)  # *2 for concatenation
        self.up2 = UpBlock(hidden_dim*2*2, hidden_dim, context_dim)    # *2 for concatenation

        
        # Final layers
        self.final = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, state_dim, 1)
        )
        
        # Condition projection
        self.cond_proj = nn.Linear(context_dim, context_dim)
    
    def forward(self, x, time_emb, context_features, n_agents):
        """
        Forward pass of the U-Net
        """
        pdb.set_trace()
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
        pdb.set_trace()
        h1 = self.down1(x, cond)
        h2 = self.down2(h1, cond)
        
        # Middle block
        h = self.mid(h2, cond)
        
        # Up blocks with skip connections
        h = self.up1(h, h2, cond)
        pdb.set_trace()
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
    def __init__(self, in_channels, out_channels, context_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.downsample = nn.AvgPool1d(2)
        
        # FiLM conditioning projection from context_dim
        self.cond_proj = nn.Linear(context_dim, out_channels*2)
    
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
    def __init__(self, channels, context_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        
        # Self-attention for sequence modeling
        self.attention = nn.MultiheadAttention(channels, 4, batch_first=True)
        
        # FiLM conditioning projection from context_dim
        self.cond_proj = nn.Linear(context_dim, channels*2)
    
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
    def __init__(self, in_channels, out_channels, context_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        
        # FiLM conditioning projection from context_dim
        self.cond_proj = nn.Linear(context_dim, out_channels*2)
    
    def forward(self, x, skip, cond):
        # Upsample
        h = self.upsample(x)
        
        # Concatenate with skip connection
        pdb.set_trace()
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

