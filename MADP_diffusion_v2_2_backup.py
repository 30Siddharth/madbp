import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from MADP_Unet_v2_2 import (
    EnhancedTrajectoryUNet, AxialPreprocessor, ContextEncoder,
    SinusoidalPosEmb, BoundaryAwareNoiseScheduler
)

class EnhancedMultiAgentDiffusionModel(nn.Module):
    def __init__(self, max_agents=10, horizon=20, state_dim=2, img_ch=3,
                 hid=128, diffusion_steps=1000, schedule_type='cosine'):
        super().__init__()
        self.max_agents = max_agents
        self.horizon = horizon
        self.state_dim = state_dim
        self.T_steps = diffusion_steps
        
        # Boundary-aware noise scheduler
        self.scheduler = BoundaryAwareNoiseScheduler(
            T=diffusion_steps,
            schedule_type=schedule_type,
            max_noise_ratio=2.5
        )
        
        # Network components
        self.time_emb = SinusoidalPosEmb(hid)
        self.ctx_enc = ContextEncoder(img_ch, pose_dim=2, hid=hid, max_agents=max_agents)
        self.preprocessor = AxialPreprocessor(
            feat_dim=hid, state_dim=2, num_heads=8,
            max_agents=max_agents, T=horizon
        )
        self.unet = EnhancedTrajectoryUNet(
            max_agents, horizon, state_dim,
            feat_dim=hid, hid_dim=hid, ctx_dim=hid
        )
    
    # -------------------------------------------------------------------------
    # Loss Computation Functions
    # -------------------------------------------------------------------------
    
    def compute_boundary_loss(self, pred_noise, target_noise, trajs, traj_noisy, starts, goals, n_agents, t):
        """Compute boundary constraint loss with proper denoising"""
        B = pred_noise.size(0)
        device = pred_noise.device
        
        # Proper denoising to get predicted x0
        alpha_t = self.scheduler.alphas_cum[t].view(-1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        # Predict x0 from noisy trajectory and predicted noise
        pred_x0 = (traj_noisy - sqrt_one_minus_alpha_t * pred_noise) / (sqrt_alpha_t + 1e-8)
        
        boundary_loss = 0.0
        
        for b in range(B):
            if isinstance(n_agents, torch.Tensor):
                n_active = n_agents[b].item()
            else:
                n_active = n_agents
            
            # Start position constraint
            start_loss = F.mse_loss(
                pred_x0[b, :n_active, 0, :],
                starts[b, :n_active, :]
            )
            
            # Goal position constraint
            goal_loss = F.mse_loss(
                pred_x0[b, :n_active, -1, :],
                goals[b, :n_active, :]
            )
            
            # Trajectory consistency loss
            traj_loss = F.mse_loss(
                pred_x0[b, :n_active],
                trajs[b, :n_active]
            )
            
            boundary_loss += start_loss + goal_loss + 0.1 * traj_loss
        
        return boundary_loss / B
    
    def compute_velocity_constraints_loss(self, pred_x0, n_agents, max_velocity=2.0, dt=0.1):
        """Enforce realistic velocity and acceleration constraints"""
        B = pred_x0.size(0)
        total_loss = 0.0
        
        for b in range(B):
            if isinstance(n_agents, torch.Tensor):
                n_active = n_agents[b].item()
            else:
                n_active = n_agents
            
            # Velocity constraints
            velocities = (pred_x0[b, :n_active, 1:, :] - pred_x0[b, :n_active, :-1, :]) / dt
            velocity_magnitudes = torch.norm(velocities, dim=-1)
            
            # Penalize excessive velocities
            velocity_violation = F.relu(velocity_magnitudes - max_velocity)
            velocity_loss = velocity_violation.mean()
            
            # Acceleration smoothness
            if velocities.size(1) > 1:
                accelerations = (velocities[:, 1:, :] - velocities[:, :-1, :]) / dt
                acceleration_loss = torch.norm(accelerations, dim=-1).mean()
                total_loss += velocity_loss + 0.1 * acceleration_loss
            else:
                total_loss += velocity_loss
        
        return total_loss / B
    
    def compute_temporal_consistency_loss(self, pred_x0, n_agents):
        """Penalize sudden changes in trajectory direction"""
        B = pred_x0.size(0)
        total_loss = 0.0
        
        for b in range(B):
            if isinstance(n_agents, torch.Tensor):
                n_active = n_agents[b].item()
            else:
                n_active = n_agents
            
            # Compute trajectory derivatives
            if pred_x0.size(2) > 2:  # Need at least 3 points
                # First derivative (velocity)
                velocities = pred_x0[b, :n_active, 1:, :] - pred_x0[b, :n_active, :-1, :]
                
                # Second derivative (acceleration)
                if velocities.size(1) > 1:
                    accelerations = velocities[:, 1:, :] - velocities[:, :-1, :]
                    
                    # Third derivative (jerk) - penalize rapid changes
                    if accelerations.size(1) > 1:
                        jerk = accelerations[:, 1:, :] - accelerations[:, :-1, :]
                        jerk_loss = torch.norm(jerk, dim=-1).mean()
                        total_loss += jerk_loss
        
        return total_loss / B
    
    def compute_goal_reaching_loss(self, pred_x0, goals, n_agents):
        """Explicit goal-reaching loss for receding horizon training"""
        B = pred_x0.size(0)
        total_loss = 0.0
        
        for b in range(B):
            if isinstance(n_agents, torch.Tensor):
                n_active = n_agents[b].item()
            else:
                n_active = n_agents
            
            # Distance from final position to goal
            final_positions = pred_x0[b, :n_active, -1, :]
            target_goals = goals[b, :n_active, :]
            
            goal_distance_loss = F.mse_loss(final_positions, target_goals)
            total_loss += goal_distance_loss
        
        return total_loss / B
    
    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------
    
    def forward(self, frames, starts, goals, n_agents, trajs, active_agents, progress):
        B = frames.size(0)
        device = frames.device
        
        # Context encoding
        ctx = self.ctx_enc(frames, starts, goals, n_agents)
        
        # Sample timesteps
        t = torch.randint(0, self.T_steps, (B,), device=device)
        
        # Forward diffusion
        x0 = trajs.permute(0, 1, 3, 2)  # [B, Na, 2, T]
        x_noisy, noise = self.scheduler.q_sample(x0, t)
        x_noisy = x_noisy.permute(0, 1, 3, 2)  # [B, Na, T, 2]
        
        # Agent masking
        agent_mask = torch.arange(self.max_agents, device=device)[None, :] >= n_agents[:, None]
        
        # Process through network
        feat = self.preprocessor(x_noisy, agent_mask)
        temb = self.time_emb(t)
        cond = ctx + temb
        pred_noise = self.unet(feat, cond)
        
        # Enhanced loss computation with Min-SNR weighting
        snr = self.scheduler.alphas_cum[t] / (1 - self.scheduler.alphas_cum[t])
        min_snr_gamma = 5.0
        weight = torch.minimum(snr, torch.tensor(min_snr_gamma, device=device)) / snr
        
        # Apply masking and weighting
        mask = (~agent_mask)[:, :, None, None].expand(-1, -1, pred_noise.size(2), pred_noise.size(3))
        weight_full = weight.view(B, 1, 1, 1).expand_as(pred_noise)
        
        weighted_pred_noise = pred_noise * weight_full
        weighted_target_noise = noise.permute(0, 1, 3, 2) * weight_full
        
        pn = weighted_pred_noise.masked_select(mask)
        nx = weighted_target_noise.masked_select(mask)
        noise_pred_loss = F.mse_loss(pn, nx)
        
        # Compute predicted x0 for additional losses
        alpha_t = self.scheduler.sqrt_ac[t].view(-1, 1, 1, 1)
        sigma_t = self.scheduler.sqrt_1mac[t].view(-1, 1, 1, 1)
        pred_x0 = (x_noisy - sigma_t * pred_noise) / (alpha_t + 1e-8)
        
        # Additional losses
        boundary_loss = self.compute_boundary_loss(
            pred_noise, noise.permute(0, 1, 3, 2), trajs, x_noisy, starts, goals, active_agents, t
        )
        velocity_loss = self.compute_velocity_constraints_loss(pred_x0, n_agents)
        temporal_loss = self.compute_temporal_consistency_loss(pred_x0, n_agents)
        
        # Progressive loss weighting
        if progress > 0.8:
            total_loss = 0.7*noise_pred_loss + 0.1*boundary_loss + 0.1*temporal_loss + 0.1*velocity_loss
        elif progress > 0.5:
            total_loss = 0.8*noise_pred_loss + 0.05*boundary_loss + 0.05*temporal_loss + 0.1*velocity_loss
        else:
            total_loss = 0.85*noise_pred_loss + 0.05*boundary_loss + 0.05*temporal_loss + 0.05*velocity_loss
        
        return total_loss
    
    # -------------------------------------------------------------------------
    # Sampling Functions
    # -------------------------------------------------------------------------
    
    @torch.no_grad()
    def sample_with_boundary_constraints(self, frames, starts, goals, n_agents, steps=50, max_step_size=0.1):
        """DDIM sampling with strict boundary constraint enforcement"""
        B = frames.size(0)
        device = frames.device
        
        # Context encoding
        ctx = self.ctx_enc(frames, starts, goals, n_agents)
        
        # Initialize with noise
        x = torch.randn(B, self.max_agents, self.horizon, 2, device=device)
        
        # Create constraint masks and values
        constraint_mask = torch.zeros(B, self.max_agents, self.horizon, device=device, dtype=torch.bool)
        constraint_values = torch.zeros(B, self.max_agents, self.horizon, 2, device=device)
        
        for b in range(B):
            if isinstance(n_agents, torch.Tensor):
                n_active = n_agents[b].item()
            else:
                n_active = n_agents
            
            # Fix start and goal positions
            constraint_mask[b, :n_active, 0] = True   # Start positions
            constraint_mask[b, :n_active, -1] = True  # Goal positions
            
            constraint_values[b, :n_active, 0] = starts[b, :n_active]
            constraint_values[b, :n_active, -1] = goals[b, :n_active]
        
        # DDIM sampling schedule
        timesteps = torch.linspace(self.T_steps-1, 0, steps).long().to(device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise
            temb = self.time_emb(t_batch)
            cond = ctx + temb
            
            agent_mask = torch.arange(self.max_agents, device=device)[None, :] >= n_agents[:, None]
            feat = self.preprocessor(x, agent_mask)
            eps = self.unet(feat, cond)
            
            # DDIM update
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t = self.scheduler.alphas_cum[t]
                alpha_next = self.scheduler.alphas_cum[t_next]
                
                # Predict x0
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
                
                # DDIM step
                x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next) * eps
            else:
                # Final denoising step
                alpha_t = self.scheduler.alphas_cum[t]
                x = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            
            # Apply velocity constraints
            if i > 0:  # Skip first iteration
                for b in range(B):
                    if isinstance(n_agents, torch.Tensor):
                        n_active = n_agents[b].item()
                    else:
                        n_active = n_agents
                    
                    # Limit step sizes between consecutive trajectory points
                    for agent_idx in range(n_active):
                        for time_idx in range(1, self.horizon):
                            step_vector = x[b, agent_idx, time_idx] - x[b, agent_idx, time_idx-1]
                            step_size = torch.norm(step_vector)
                            
                            if step_size > max_step_size:
                                # Scale down the step to maximum allowed size
                                x[b, agent_idx, time_idx] = (
                                    x[b, agent_idx, time_idx-1] +
                                    step_vector * (max_step_size / step_size)
                                )
            
            # CRITICAL: Apply boundary constraints after each step
            constraint_mask_expanded = constraint_mask.unsqueeze(-1).expand_as(x)
            constraint_values_expanded = constraint_values
            x[constraint_mask_expanded] = constraint_values_expanded[constraint_mask_expanded]
        
        return x
    
    @torch.no_grad()
    def sample_full_trajectory_receding_horizon(self, frames, starts, goals, n_agents, 
                                              full_horizon_length, horizon_size=10, max_step_size=0.1):
        """Generate full trajectory using receding horizon with boundary constraints"""
        B = frames.size(0)
        device = frames.device
        
        # Initialize full trajectory tensor
        full_trajectory = torch.zeros(B, self.max_agents, full_horizon_length, 2, device=device)
        current_positions = starts.clone()
        
        # Calculate number of horizons needed
        num_horizons = full_horizon_length // horizon_size
        
        for horizon_idx in range(num_horizons):
            start_t = horizon_idx * horizon_size
            end_t = start_t + horizon_size
            
            # Set intermediate goals for non-final horizons
            if horizon_idx == num_horizons - 1:
                # Final horizon - use actual goals
                horizon_goals = goals
            else:
                # Intermediate horizon - interpolate toward final goals
                progress = (horizon_idx + 1) / num_horizons
                horizon_goals = current_positions + progress * (goals - current_positions)
            
            # Sample current horizon with boundary constraints
            horizon_prediction = self.sample_with_boundary_constraints(
                frames, current_positions, horizon_goals, n_agents, 
                steps=50, max_step_size=max_step_size
            )
            
            # Store in full trajectory
            full_trajectory[:, :, start_t:end_t, :] = horizon_prediction
            
            # Update current positions for next horizon
            current_positions = horizon_prediction[:, :, -1, :].clone()
        
        return full_trajectory
    
    def compute_receding_horizon_goal_loss(self, pred_x0, goals, n_agents):
        """Goal-reaching loss for receding horizon training"""
        return self.compute_goal_reaching_loss(pred_x0, goals, n_agents)
