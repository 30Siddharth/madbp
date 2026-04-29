import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import os
import glob

from Traj_UNet_modified import TrajectoryUNet

# ===== Multi-Agent Diffusion Policy =====

class MultiAgentDiffusionPolicy(nn.Module):
    def __init__(
        self,
        state_dim=6,
        image_channels=3,
        context_dim=64,
        max_num_agents=4,
        horizon=49,
        hidden_dim=128,
        n_diffusion_steps=100,
        beta_schedule="constant",
        noise=None
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
        elif beta_schedule == "constant":
            # Adjust this value based on your needs
            constant_value = 1e-3
            betas = torch.ones(n_diffusion_steps) * constant_value
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
            nn.Linear(4 * max_num_agents, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Goal State Encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(2 * max_num_agents, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Agent encoder
        self.agent_count_encoder = nn.Embedding(max_num_agents + 1, hidden_dim // 4)
        
        # Add cross-attention between context elements
        self.context_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Enhanced U-Net for trajectory denoising with agent attention
        self.trajectory_unet = TrajectoryUNet(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            context_dim=hidden_dim,
            max_num_agents=max_num_agents
        )
        
        # Agent-specific context encoder
        self.agent_context_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),  # 4 for agent state
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def process_context(self, context):
        '''
        Process the heterogenous context tokens.
        '''
        # Unpack context
        images, start_states, goal_states, num_agents = context
        
        # Handle time dimension in images
        b, _, c, h, w = images.shape
        img_features = self.image_encoder(images[:, 0])
        
        # Handle start states (B, Na, 4)
        start_features = self.state_encoder(start_states.reshape(b, -1))
        
        # Handle goal states (B, Na, 2)
        goal_features = self.goal_encoder(goal_states.reshape(b, -1))
        
        # Use the number of agents directly
        agent_features = self.agent_count_encoder(num_agents).squeeze(1)
        
        # Combine features
        combined = torch.cat([img_features, start_features, goal_features, agent_features], dim=1)
        
        # Apply attention for better integration
        context_features = self.context_attention(combined, combined, combined)[0]
        
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
    
    def forward_diffusion_per_agent(self, x_start, t, active_agents=None, noise=None):
        """
        Forward diffusion process that preserves the agent dimension: q(x_k | x_0)
        x_start shape: [B, T, Na, D]
        active_agents: tensor of shape [B] indicating number of active agents for each batch item
        """
        B, T, Na, D = x_start.shape
        
        # Create agent indicators
        agent_mask = None
        if active_agents is not None:
            agent_mask = torch.zeros((B, Na), device=x_start.device)
            for b in range(B):
                agent_mask[b, :active_agents[b]] = 1.0
        
        if noise is None:
            # Generate independent noise for each agent
            noise = torch.randn_like(x_start)
        elif noise == 'constrained':
            noise = torch.rand_like(x_start) - x_start
        
        # Reshape t for broadcasting
        k_idx = t.view(-1, 1, 1, 1)
        
        # Get diffusion parameters for the timestep
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[k_idx]
        sqrt_one_minus_alphas_cumprod_k = self.sqrt_one_minus_alphas_cumprod[k_idx]
        
        # Apply diffusion to each agent independently
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_k * noise
        
        return x_noisy, noise, agent_mask
    
    def p_losses(self, x_start, context, t, active_agents=None, noise=None, loss_type="cw"):
        """
        Compute loss for the diffusion model with agent-independent noise
        active_agents: tensor of shape [B] indicating number of active agents for each batch item
        """
        B, T, max_agents, D = x_start.shape
        
        # If active_agents not provided, use all agents
        if active_agents is None:
            active_agents = torch.full((B,), max_agents, device=x_start.device, dtype=torch.long)
        
        # 1. Forward diffusion process with agent-independent noise
        x_noisy, noise, agent_mask = self.forward_diffusion_per_agent(x_start, t, active_agents, noise)
        
        # 2. Process context
        context_features = self.process_context(context)
        time_emb = self.get_time_embedding(t, hidden_dim=context_features.shape[-1])
        
        # 3. Predict noise with U-Net
        predicted_noise = self.trajectory_unet(x_noisy, time_emb, context_features, active_agents)
        
        # 4. Compute loss
        if loss_type in ['l1', 'l2', 'huber']:
            # Create a mask for active agents
            agent_mask = torch.zeros((B, T, max_agents, D), device=x_start.device)
            for b in range(B):
                agent_mask[b, :, :active_agents[b], :] = 1.0
                
            if loss_type == 'l1':
                loss = F.l1_loss(noise * agent_mask, predicted_noise * agent_mask, reduction='sum')
            elif loss_type == 'l2':
                loss = F.mse_loss(noise * agent_mask, predicted_noise * agent_mask, reduction='sum')
            elif loss_type == "huber":
                loss = F.smooth_l1_loss(noise * agent_mask, predicted_noise * agent_mask, reduction='sum')
                
            # Normalize by the number of active elements
            active_elements = agent_mask.sum()
            loss = loss / active_elements if active_elements > 0 else loss
            
        elif loss_type == 'cw':
            # Reconstruct clean trajectory from predicted noise
            pred_x0 = self.predict_start_from_noise(x_noisy, t, predicted_noise)
            
            # Create a mask for active agents
            agent_mask = torch.zeros((B, T, max_agents, D), device=x_start.device)
            for b in range(B):
                agent_mask[b, :, :active_agents[b], :] = 1.0
                
            # Compute component loss on actual trajectory components with masking
            loss = self.component_weighted_loss(pred_x0, x_start, agent_mask)
            
        else:
            raise NotImplementedError()
            
        return loss
    
    def predict_start_from_noise(self, x_k, t, predicted_noise):
        """Convert noise prediction to clean trajectory using Tweedie's formula"""
        k_idx = t.view(-1, 1, 1, 1)
        sqrt_recip_alphas_k = self.sqrt_recip_alphas[k_idx]
        sqrt_one_minus_alphas_cumprod_k = self.sqrt_one_minus_alphas_cumprod[k_idx]
        
        return (x_k - sqrt_one_minus_alphas_cumprod_k * predicted_noise) / sqrt_recip_alphas_k
    
    def component_weighted_loss(self, pred_x0, true_x0, mask=None):
        """
        Weight position > velocity > action with optional masking for active agents
        """
        weights = [0.0, 0.0, 1.0]  # Position, velocity, action weights
        
        if mask is None:
            pos_loss = F.mse_loss(pred_x0[..., :2], true_x0[..., :2])
            vel_loss = F.mse_loss(pred_x0[..., 2:4], true_x0[..., 2:4])
            act_loss = F.mse_loss(pred_x0[..., 4:], true_x0[..., 4:])
        else:
            # Apply mask to each component
            pos_mask = mask[..., :2]
            vel_mask = mask[..., 2:4]
            act_mask = mask[..., 4:]
            
            # Compute masked losses
            pos_diff = ((pred_x0[..., :2] - true_x0[..., :2]) ** 2) * pos_mask
            vel_diff = ((pred_x0[..., 2:4] - true_x0[..., 2:4]) ** 2) * vel_mask
            act_diff = ((pred_x0[..., 4:] - true_x0[..., 4:]) ** 2) * act_mask
            
            # Normalize by the number of active elements
            pos_active = pos_mask.sum()
            vel_active = vel_mask.sum()
            act_active = act_mask.sum()
            
            pos_loss = pos_diff.sum() / pos_active if pos_active > 0 else 0
            vel_loss = vel_diff.sum() / vel_active if vel_active > 0 else 0
            act_loss = act_diff.sum() / act_active if act_active > 0 else 0
            
        return weights[0] * pos_loss + weights[1] * vel_loss + weights[2] * act_loss
    
    def forward(self, x, context, active_agents=None):
        """
        Forward pass during training with curriculum learning for number of agents
        active_agents: tensor of shape [B] indicating number of active agents for each batch item
        """
        b, *_ = x.shape
        t = torch.randint(0, self.n_diffusion_steps, (b,), device=x.device).long()
        return self.p_losses(x, context, t, active_agents)
    
    @torch.no_grad()
    def p_sample(self, x, context_features, k, k_index, active_agents):
        """
        Sample from the model at timestep t (denoising step) with agent-independent processing
        active_agents: tensor of shape [B] indicating number of active agents for each batch item
        """
        B, T, max_agents, D = x.shape
        
        # Reshape t for broadcasting
        k_idx = k.view(-1, 1, 1, 1)
        
        # Get diffusion parameters for the timestep
        betas_k = self.betas[k_idx]
        sqrt_one_minus_alphas_cumprod_k = self.sqrt_one_minus_alphas_cumprod[k_idx]
        sqrt_recip_alphas_k = self.sqrt_recip_alphas[k_idx]
        
        # Get time embedding
        time_emb = self.get_time_embedding(k, hidden_dim=context_features.shape[-1])
        
        # Predict noise with U-Net (preserving agent dimension)
        predicted_noise = self.trajectory_unet(x, time_emb, context_features, active_agents)
        
        # Equation 11 in the DDPM paper
        # Use our model (noise predictor) to predict the mean
        mean = sqrt_recip_alphas_k * (x - betas_k * predicted_noise / sqrt_one_minus_alphas_cumprod_k)
        
        if k_index == 0:
            return mean
        else:
            posterior_variance_k = self.posterior_variance[k_idx]
            
            # Generate independent noise for each agent
            noise = torch.randn_like(x)
            
            # Create a mask for active agents
            agent_mask = torch.zeros((B, T, max_agents, D), device=x.device)
            for b in range(B):
                agent_mask[b, :, :active_agents[b], :] = 1.0
                
            # Apply noise only to active agents
            noised_mean = mean + torch.sqrt(posterior_variance_k) * noise * agent_mask
            
            return noised_mean
    
    @torch.no_grad()
    def sample_trajectory(self, context, active_agents=None, shape=None, save_steps=True, step_interval=50):
        """
        Sample a trajectory from the model (inference) with curriculum learning for number of agents
        active_agents: tensor of shape [B] indicating number of active agents for each batch item
        """
        if shape is None:
            batch_size = context[0].shape[0]  # Get batch size from first context element
            max_agents = self.max_num_agents
            
            # If active_agents not provided, use max_num_agents for all batch items
            if active_agents is None:
                active_agents = torch.full((batch_size,), max_agents,
                                          device=next(self.parameters()).device,
                                          dtype=torch.long)
                
            shape = (batch_size, self.horizon, max_agents, self.state_dim)
            
        device = next(self.parameters()).device
        
        # Start from pure noise for all agents
        x = torch.randn(shape, device=device)
        
        # Get context features once
        context_features = self.process_context(context)
        
        # For storing intermediate results
        intermediate_trajectories = {}
        
        # Iterative denoising process
        for i in reversed(range(0, self.n_diffusion_steps)):
            k = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, context_features, k, i, active_agents)
            
            # Save intermediate trajectory at specified intervals
            if save_steps and (i % step_interval == 0 or i == self.n_diffusion_steps - 1):
                intermediate_trajectories[i] = x.clone().detach().cpu()
                
        if save_steps:
            return x, intermediate_trajectories
        else:
            return x
