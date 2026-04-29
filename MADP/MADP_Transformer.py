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
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import seaborn as sns

device = 'cuda:0' if torch.cuda.is_available() else "cpu"

class MultiAgentDiffusionPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int = 6,
        image_channels: int = 3,
        context_dim: int = 64,
        max_num_agents: int = 4,
        horizon: int = 49,
        hidden_dim: int = 128,  # Reduced from 256
        n_diffusion_steps: int = 100,
        beta_schedule: str = "linear",
        noise: str = None
    ):
        super().__init__()
        self.state_dim = state_dim
        self.max_num_agents = max_num_agents
        self.horizon = horizon
        self.n_diffusion_steps = n_diffusion_steps
        self.noise = noise
        
        # Define betas for diffusion process (noise scheduler)
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 1e-2, n_diffusion_steps)
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

        self.image_encoder_ = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
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
        self.trajectory_unet = TrajectoryTransformer(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            context_dim=hidden_dim,
            max_num_agents=max_num_agents
        )
    
    def process_context(self, context):
        '''
        Process the heterogenous context tokens.
        '''
        # Unpack context
        # TODO: Update this unpaking for dictionary/hash-map based context.
        images, start_states, goal_states, num_agents = context
        
        # Handle time dimension in images (B, T, C, H, W)
        # Only use the first frame (t=0) from images (B, T, C, H, W)
        # b, _, c, h, w = images.shape
        b,_, c,h,w = images.shape
        # images_t0 = images[:, 0]  # Shape: (B, C, H, W)
        # img_features = self.image_encoder(images_t0)
        
        
        img_features = self.image_encoder(images[:, 0] )
        
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
        
        img_features_ = self.image_encoder_(images[:, 0] )
        pdb.set_trace()
        features_np = img_features.detach().cpu().numpy()

        plt.figure(figsize=(12, 10))
        sns.heatmap(features_np, cmap='viridis')
        plt.title('Context Features Heatmap')
        plt.savefig('context_features_heatmap.png')
        plt.show()
        
        pdb.set_trace()

        n_row = b
        images_tensor = images.reshape(b,c,h,w)
        title="Images"
        grid = vutils.make_grid(images_tensor, nrow=n_row, padding=10, normalize=True)
        plt.figure(figsize=(15, 15))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        if title:
            plt.title(title)
        plt.axis("off")
        plt.show()

        return context_features
    
    def plot_image_tensor(tensor, nrow=8, padding=2, normalize=True, title=None):
        """
        Plots a tensor of images.

        Args:
            tensor (Tensor): A 4D tensor of shape (B, C, H, W)
            nrow (int): Number of images per row
            padding (int): Padding between images
            normalize (bool): Whether to normalize images to [0,1]
            title (str): Optional title
        """
        grid = vutils.make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize)
        plt.figure(figsize=(15, 15))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        if title:
            plt.title(title)
        plt.axis("off")
        plt.show()
    
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
        Forward diffusion process with constrained action noise
        """
        if noise is None:
            # Generate standard noise for all dimensions
            noise = torch.randn_like(x_start)
            
            # Apply scaling to action noise based on proximity to boundaries
            actions = x_start[..., 4:6]
            distance_to_boundary = 1.0 - torch.abs(actions)
            scaling_factor = torch.clamp(distance_to_boundary, 0.1, 1.0)
            
            # Scale noise for actions
            noise[..., 4:6] = noise[..., 4:6] * scaling_factor
        
        # Apply diffusion
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # Calculate noisy sample
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Ensure actions stay within bounds even during diffusion
        x_max = torch.tensor([2,2], device=device)
        x_t[..., :2] = torch.clamp(x_t[..., :2], -x_max, x_max)
        # x_t[..., 2:4] = torch.clamp(x_t[..., 2:4], -v_max, v_max)
        x_t[..., 4:6] = torch.clamp(x_t[..., 4:6], -0.5, 0.5)
        
        return x_t, noise

    
    def p_losses(self, x_start, context, t, noise=None, loss_type="cw"):
        B, T, n_agents, D = x_start.shape
        
        # 1. Forward diffusion process
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy, noise = self.forward_diffusion(x_start, t, noise)
        
        # 2. Predict noise with U-Net
        context_features = self.process_context(context)
        time_emb = self.get_time_embedding(t, hidden_dim=context_features.shape[-1])
        predicted_noise = self.trajectory_unet(x_noisy, time_emb, context_features, n_agents)
        
        # 3. Base noise prediction loss
        if loss_type in ['l1', 'l2', 'huber']:
            if loss_type == 'l1':
                loss = F.l1_loss(noise, predicted_noise)
            elif loss_type == 'l2':
                loss = F.mse_loss(noise, predicted_noise)
            elif loss_type == "huber":
                loss = F.smooth_l1_loss(noise, predicted_noise)
        
        # 4. Component-weighted trajectory loss
        elif loss_type == 'cw':
            # Reconstruct clean trajectory from predicted noise
            pred_x0 = self.predict_start_from_noise(x_noisy, t, predicted_noise)
            
            # Compute component loss on actual trajectory components
            loss = self.component_weighted_loss(pred_x0, x_start)
        
        else:
            raise NotImplementedError()
        
        return loss

    def predict_start_from_noise(self, x_t, t, predicted_noise):
        """Convert noise prediction to clean trajectory using Tweedie's formula"""
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_recip_alphas_t


    def component_weighted_loss(self, pred_x0, true_x0):
        weights = [0.01,
                   0.00,
                   1.0]
        """Weight position > velocity > action"""
        pos_loss = F.mse_loss(pred_x0[..., :2], true_x0[..., :2])
        vel_loss = F.mse_loss(pred_x0[..., 2:4], true_x0[..., 2:4])
        act_loss = F.mse_loss(pred_x0[..., 4:], true_x0[..., 4:])
        return weights[0]*pos_loss + weights[1]*vel_loss + weights[2]*act_loss  # Adjust weights as needed

    
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
        Sample from the model at timestep t with action constraints
        """
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        
        # Get time embedding
        time_emb = self.get_time_embedding(t, hidden_dim=context_features.shape[-1])
        
        # Predict noise with U-Net
        predicted_noise = self.trajectory_unet(x, time_emb, context_features, n_agents)
        
        # Predict the mean
        mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        # Apply action constraints
        mean[..., 4:6] = torch.clamp(mean[..., 4:6], -0.5, 0.5)

        x_max = torch.tensor([2,2], device=device)
        if t_index == 0:
            # Clamp state dimensions (e.g., position: [:2], velocity: [2:4])
            mean[..., :2] = torch.clamp(mean[..., :2], -x_max, x_max)         # Position
            # mean[..., 2:4] = torch.clamp(mean[..., 2:4], -v_max, v_max)       # Velocity
            return mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            noisy_sample = mean + torch.sqrt(posterior_variance_t) * noise
            # Clamp state dimensions after adding noise
            noisy_sample[..., :2] = torch.clamp(noisy_sample[..., :2], -x_max, x_max)
            # sample[..., 2:4] = torch.clamp(sample[..., 2:4], -v_max, v_max)
            return noisy_sample

    
    @torch.no_grad()
    def sample_trajectory(self, context, n_agents, shape=None, save_steps=True, step_interval=50):
        """
        Sample a trajectory from the model (inference)
        
        Args:
            context: Context information (images, start states, goal states, num_agents)
            n_agents: Number of agents
            shape: Shape of the trajectory to generate
            save_steps: Whether to save intermediate steps
            step_interval: Interval at which to save steps
            
        Returns:
            x: Final denoised trajectory
            intermediate_trajectories: Dict mapping step indices to trajectories (if save_steps=True)
        """
        if shape is None:
            batch_size = context[0].shape[0]  # Get batch size from first context element
            shape = (batch_size, self.horizon, n_agents, self.state_dim)
        
        device = next(self.parameters()).device
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Get context features once
        context_features = self.process_context(context)
        
        # For storing intermediate results
        intermediate_trajectories = {}
        
        # Iterative denoising process
        for i in reversed(range(0, self.n_diffusion_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, context_features, t, i, n_agents)
            
            # Save intermediate trajectory at specified intervals
            if save_steps and (i % step_interval == 0 or i == self.n_diffusion_steps - 1):
                intermediate_trajectories[i] = x.clone().detach().cpu()
        
        if save_steps:
            return x, intermediate_trajectories
        else:
            return x


class TrajectoryTransformer(nn.Module):
    def __init__(self, state_dim, hidden_dim, context_dim, max_num_agents):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.proj = nn.Linear(state_dim, hidden_dim)
        
        # Time and context conditioning
        self.cond_proj = nn.Linear(context_dim, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim*2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Final projection
        self.final = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, x, time_emb, context_features, n_agents):
        batch_size, horizon, _, _ = x.shape
        
        # Reshape for transformer: (batch_size * n_agents, horizon, state_dim)
        x = x[:, :, :n_agents, :].reshape(batch_size * n_agents, horizon, self.state_dim)
        
        # Initial projection
        h = self.proj(x)  # -&gt; (batch_size * n_agents, horizon, hidden_dim)
        
        # Add time and context conditioning
        # pdb.set_trace()
        cond = self.cond_proj(time_emb + context_features)
        cond = cond.unsqueeze(1).repeat(1, horizon, 1)  # Expand to all timesteps
        cond = cond.repeat_interleave(n_agents, dim=0)  # Repeat for each agent
        
        # Add conditioning to input
        h = h + cond
        
        # Apply transformer
        h = self.transformer(h)
        
        # Final projection
        output = self.final(h)  # -&gt; (batch_size * n_agents, horizon, state_dim)
        
        # Reshape back
        output = output.reshape(batch_size, n_agents, horizon, self.state_dim)
        output = output.permute(0, 2, 1, 3)  # -&gt; (batch_size, horizon, n_agents, state_dim)
        
        return output