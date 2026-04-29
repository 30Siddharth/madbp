import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.interpolate import CubicSpline
import numpy as np
import pdb

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

        # Enhanced scheduler with bounded noise to prevent scaling issues
        self.scheduler = BoundaryAwareNoiseScheduler(
            T=diffusion_steps,
            schedule_type=schedule_type,
            max_noise_ratio=3.0
        )

        self.time_emb = SinusoidalPosEmb(hid)
        self.ctx_enc = ContextEncoder(img_ch, pose_dim=2, hid=hid, max_agents=max_agents)
        self.preprocessor = AxialPreprocessor(
            feat_dim=hid, state_dim=2, num_heads=4,
            max_agents=max_agents, T=horizon
        )
        self.input_proj = nn.Linear(state_dim, hid)  # To test without Axial Processor

        self.unet = EnhancedTrajectoryUNet(
            max_agents, horizon, state_dim,
            feat_dim=hid, hid_dim=hid, ctx_dim=hid
        )

    def compute_boundary_loss(self, pred_noise, target_noise, trajs, traj_noisy, starts, goals, n_agents, t):
        """Compute additional loss to enforce boundary constraints"""
        B = pred_noise.size(0)
        device = pred_noise.device

        # Proper denoising step instead of simple subtraction
        alpha_t = self.scheduler.alphas_cum[t].view(-1, 1, 1, 1)
        pred_x0 = (traj_noisy - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)

        boundary_loss = 0.0

        for b in range(B):
            # FIXED: Handle both tensor and int n_agents
            if isinstance(n_agents, torch.Tensor):
                n_active = n_agents[b].item()
            else:
                n_active = n_agents

            # Start position loss
            start_loss = F.mse_loss(
                pred_x0[b, :n_active, 0, :],
                starts[b, :n_active, :]
            )

            # Goal position loss
            goal_loss = F.mse_loss(
                pred_x0[b, :n_active, -1, :],
                goals[b, :n_active, :]
            )

            # Traj loss
            traj_loss = F.mse_loss(pred_x0[b, :n_active], trajs[b, :n_active])

            boundary_loss += start_loss + goal_loss + 0.1*traj_loss

        return boundary_loss / B

    def compute_trajectory_derivatives(self, trajs):
        """
        Compute velocity and acceleration from trajectories
        Args:
            trajs: Trajectory tensor [B, Na, T, 2] (x, y positions)
        Returns:
            velocities: [B, Na, T-1, 2]
            accelerations: [B, Na, T-2, 2]
        """
        # Velocity: finite difference between consecutive positions
        velocities = trajs[:, :, 1:, :] - trajs[:, :, :-1, :]
        # Acceleration: finite difference between consecutive velocities
        accelerations = velocities[:, :, 1:, :] - velocities[:, :, :-1, :]
        return velocities, accelerations

    def compute_temporal_consistency_loss(self, pred_x0, n_agents):
        """Penalize sudden changes in trajectory direction"""
        pred_velocities, pred_accelerations = self.compute_trajectory_derivatives(pred_x0)
        total_loss = 0.0

        for b in range(pred_x0.size(0)):
            if isinstance(n_agents, torch.Tensor):
                n_active = n_agents[b].item()
            else:
                n_active = n_agents

            # Penalize large changes in velocity direction
            if pred_velocities.size(2) > 1:  # Need at least 2 velocity vectors
                velocity_changes = pred_velocities[b, :n_active, 1:] - pred_velocities[b, :n_active, :-1]
                direction_change_loss = torch.norm(velocity_changes, dim=-1).mean()
                total_loss += direction_change_loss

            # Penalize large accelerations (jerk minimization)
            if pred_accelerations.size(2) > 1:
                jerk = pred_accelerations[b, :n_active, 1:] - pred_accelerations[b, :n_active, :-1]
                jerk_loss = torch.norm(jerk, dim=-1).mean()
                total_loss += 0.1 * jerk_loss

        return total_loss / pred_x0.size(0)

    def compute_velocity_constraints_loss(self, pred_x0, n_agents, max_velocity=2.0, dt=0.1):
        """Enforce realistic velocity and acceleration constraints"""
        B = pred_x0.size(0)
        total_loss = 0.0

        for b in range(B):
            if isinstance(n_agents, torch.Tensor):
                n_active = n_agents[b].item()
            else:
                n_active = n_agents

            # Velocity constraints (limit step size between consecutive points)
            velocities = (pred_x0[b, :n_active, 1:, :] - pred_x0[b, :n_active, :-1, :]) / dt
            velocity_magnitudes = torch.norm(velocities, dim=-1)

            # Penalize velocities exceeding maximum
            velocity_violation = F.relu(velocity_magnitudes - max_velocity)
            velocity_loss = velocity_violation.mean()

            # Acceleration constraints (smooth movement)
            if velocities.size(1) > 1:
                accelerations = (velocities[:, 1:, :] - velocities[:, :-1, :]) / dt
                acceleration_loss = torch.norm(accelerations, dim=-1).mean()
                total_loss += velocity_loss + 0.1 * acceleration_loss
            else:
                total_loss += velocity_loss

        return total_loss / B

    def compute_collision_loss(self, pred_x0, n_agents, min_distance=0.1):
        """Penalize trajectories with inter-agent collisions"""
        B = pred_x0.size(0)
        total_loss = 0.0

        for b in range(B):
            if isinstance(n_agents, torch.Tensor):
                n_active = n_agents[b].item()
            else:
                n_active = n_agents

            # Compute pairwise distances at each timestep
            for t in range(pred_x0.size(2)):
                positions = pred_x0[b, :n_active, t, :]  # [n_active, 2]
                # Pairwise distance matrix
                dist_matrix = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0))[0]
                # Mask diagonal (self-distances)
                mask = ~torch.eye(n_active, device=positions.device, dtype=torch.bool)
                distances = dist_matrix[mask]
                # Penalize distances below minimum threshold
                collision_penalty = F.relu(min_distance - distances).sum()
                total_loss += collision_penalty

        return total_loss / B

    def compute_receding_horizon_goal_loss(self, pred_x0, goals, n_agents):
        """Compute loss for staying away from final goal at end of each horizon"""
        B = pred_x0.size(0)
        total_loss = 0.0

        for b in range(B):
            if isinstance(n_agents, torch.Tensor):
                n_active = n_agents[b].item()
            else:
                n_active = n_agents

            # Distance from final position to ultimate goal
            final_positions = pred_x0[b, :n_active, -1, :]
            target_goals = goals[b, :n_active, :]
            goal_distance_loss = F.mse_loss(final_positions, target_goals)
            total_loss += goal_distance_loss

        return total_loss / B
    
    def compute_collision_loss(self, pred_x0, n_agents, min_distance=0.1):
        """Penalize trajectories with inter-agent collisions"""
        B = pred_x0.size(0)
        total_loss = 0.0
        
        for b in range(B):
            n_active = n_agents[b].item() if isinstance(n_agents, torch.Tensor) else n_agents
            
            # Compute pairwise distances at each timestep
            for t in range(pred_x0.size(2)):
                positions = pred_x0[b, :n_active, t, :]  # [n_active, 2]
                
                # Pairwise distance matrix
                dist_matrix = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0))[0]
                
                # Mask diagonal (self-distances)
                mask = ~torch.eye(n_active, device=positions.device, dtype=torch.bool)
                distances = dist_matrix[mask]
                
                # Penalize distances below minimum threshold
                collision_penalty = F.relu(min_distance - distances).sum()
                total_loss += collision_penalty
        
        return total_loss / B

    def forward(self, frames, starts, goals, n_agents, trajs, active_agents, progress):
        B = frames.size(0)
        device = frames.device
        # pdb.set_trace()

        # Context encoding
        ctx = self.ctx_enc(frames, starts, goals, n_agents)

        # Limited timestep sampling for better training stability
        t_min = 0
        t_max = self.T_steps
        t = torch.randint(t_min, t_max, (B,), device=device)

        # Forward diffusion with proper tensor handling
        x0 = trajs.permute(0, 1, 3, 2)  # [B, Na, 2, T]
        x_noisy, noise = self.scheduler.q_sample(x0, t)
        x_noisy = x_noisy.permute(0, 1, 3, 2)  # [B, Na, T, 2]

        # Agent masking - FIXED: Use max_agents consistently
        agent_mask = torch.arange(self.max_agents, device=device)[None, :] >= n_agents[:, None]

        # Process through network
        # feat = self.preprocessor(x_noisy, agent_mask)

        feat = self.input_proj(x_noisy)
        # feat = x_noisy
        temb = self.time_emb(t)
        cond = ctx + temb

        pred_noise = self.unet(feat, cond)

        # Enhanced loss with Min-SNR weighting for better convergence
        snr = self.scheduler.alphas_cum[t] / (1 - self.scheduler.alphas_cum[t])
        min_snr_gamma = 5.0
        weight = torch.minimum(snr, torch.tensor(min_snr_gamma, device=device)) / snr

        # Apply weighting before masking
        mask = (~agent_mask)[:, :, None, None].expand(-1, -1, pred_noise.size(2), pred_noise.size(3))
        # Expand weight to match the full tensor dimensions
        weight_full = weight.view(B, 1, 1, 1).expand_as(pred_noise)

        # Apply weighting and then mask
        weighted_pred_noise = pred_noise * weight_full
        weighted_target_noise = noise.permute(0, 1, 3, 2) * weight_full

        # Apply mask and compute loss
        pn = weighted_pred_noise.masked_select(mask)
        nx = weighted_target_noise.masked_select(mask)
        noise_pred_loss = F.mse_loss(pn, nx)

        alpha_t = self.scheduler.sqrt_ac[t].view(-1, 1, 1, 1)
        sigma_t = self.scheduler.sqrt_1mac[t].view(-1, 1, 1, 1)
        pred_x0 = (x_noisy - sigma_t * pred_noise) / (alpha_t + 1e-8)

        # Compute loss for boundary conditions
        boundary_loss = self.compute_boundary_loss(pred_noise, noise.permute(0, 1, 3, 2), trajs, x_noisy, starts, goals, n_agents, t)

        # Loss for velocity and acceleration
        velocity_loss = self.compute_velocity_constraints_loss(pred_x0, n_agents, max_velocity=2.0, dt=0.1)

        # Temporal consistency loss
        temporal_loss = self.compute_temporal_consistency_loss(pred_x0, n_agents)

        # Loss after each horizon for missing goal
        horizon_loss = self.compute_receding_horizon_goal_loss(pred_x0, goals, n_agents)

        # Loss based on collision
        collision_loss = self.compute_collision_loss(pred_x0, n_agents)

        if progress > 0.8:
            total_loss = 0.8*noise_pred_loss + 0.00*boundary_loss + 0.1*temporal_loss + 0.1*velocity_loss
        elif progress > 0.5:
            total_loss = 0.9*noise_pred_loss + 0.00*boundary_loss + 0.0*temporal_loss + 0.1*velocity_loss
        else:
            total_loss = 0.9*noise_pred_loss + 0.0*boundary_loss + 0.01*temporal_loss + 0.09*velocity_loss

        return 0.9*noise_pred_loss + 0.06*temporal_loss + 0.04*boundary_loss + 0.0*collision_loss

    @torch.no_grad()
    def sample_with_constraints(self, frames, starts, goals, n_agents, steps=50, max_step_size=0.1, dataset=None):
        """DDIM sampling with start/goal constraint enforcement"""
        B = frames.size(0)
        device = frames.device

        ctx = self.ctx_enc(frames, starts, goals, n_agents)
        x = torch.randn(B, self.max_agents, self.horizon, 2, device=device)

        # Create constraint mask for start and goal positions
        constraint_mask = torch.zeros(B, self.max_agents, self.horizon, device=device, dtype=torch.bool)
        constraint_mask[:, :, 0] = True  # Fix start positions
        constraint_mask[:, :, -1] = True  # Fix goal positions

        # Prepare constraint values
        constraint_values = torch.zeros(B, self.max_agents, self.horizon, 2, device=device)
        for b in range(B):
            if isinstance(n_agents, torch.Tensor):
                n_active = n_agents[b].item()
            else:
                n_active = n_agents
            constraint_values[b, :n_active, 0] = starts[b, :n_active]  # Start positions
            constraint_values[b, :n_active, -1] = goals[b, :n_active]  # Goal positions

        t_min = 0
        t_max = self.T_steps
        timesteps = torch.linspace(t_max-1, 0, steps).long().to(device)

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

                pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
                x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next) * eps
            else:
                alpha_t = self.scheduler.alphas_cum[t]
                x = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)

            max_step_normalized = max_step_size

            if i > 0:  # Skip first iteration
                for b in range(B):
                    if isinstance(n_agents, torch.Tensor):
                        n_active = n_agents[b].item()
                    else:
                        n_active = n_agents

                    # Limit step sizes between consecutive trajectory points
                    for agent_idx in range(n_active):
                        for time_idx in range(1, self.horizon):
                            if time_idx == self.horizon - 1:
                                continue
                            step_vector = x[b, agent_idx, time_idx] - x[b, agent_idx, time_idx-1]
                            step_size = torch.norm(step_vector)
                            if step_size > max_step_normalized:
                                # Scale down the step to maximum allowed size
                                x[b, agent_idx, time_idx] = (
                                    x[b, agent_idx, time_idx-1] +
                                    step_vector * (max_step_normalized / step_size)
                                )
            # import pdb
            # pdb.set_trace()
            # CRITICAL: Apply constraints after each denoising step
            x[constraint_mask.unsqueeze(-1).expand_as(x)] = constraint_values[constraint_mask.unsqueeze(-1).expand_as(constraint_values)]

        return x

    @torch.no_grad()
    def sample_full_trajectory_receding_horizon(self, frames, starts, goals, n_agents,
                                              full_horizon_length, horizon_size=10,
                                              max_step_size=0.1):
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
                horizon_goals = goals
            else:
                # horizon_goals = goals
                progress = (horizon_idx + 1) / num_horizons
                horizon_goals = current_positions + progress * (goals - current_positions)

            # Sample and smooth current horizon
            horizon_prediction = self.sample_with_constraints(
                frames, current_positions, horizon_goals, n_agents,
                steps=50, max_step_size=max_step_size
            )

            # Store in full trajectory
            full_trajectory[:, :, start_t:end_t, :] = horizon_prediction

            # Update current positions for next horizon
            current_positions = horizon_prediction[:, :, -1, :].clone()

        return full_trajectory
