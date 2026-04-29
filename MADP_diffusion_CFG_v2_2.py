import torch
import torch.nn as nn
import torch.nn.functional as F
from MADP_diffusion_v2_2 import EnhancedMultiAgentDiffusionModel

class CFGEnhancedMultiAgentDiffusionModel(EnhancedMultiAgentDiffusionModel):
    def __init__(self, max_agents=10, horizon=20, state_dim=2, img_ch=3,
                 hid=128, diffusion_steps=1000, schedule_type='cosine',
                 cfg_dropout_prob=0.1):
        super().__init__(max_agents, horizon, state_dim, img_ch, hid, 
                        diffusion_steps, schedule_type)
        
        # CFG-specific parameters
        self.cfg_dropout_prob = cfg_dropout_prob
        
    def create_unconditional_context(self, batch_size, device):
        """Create unconditional context by zeroing out conditioning information"""
        return torch.zeros(batch_size, self.ctx_enc.fusion.embed_dim, device=device)
    
    def forward(self, frames, starts, goals, n_agents, trajs, active_agents, progress):
        """Enhanced forward pass with CFG conditional dropout"""
        B = frames.size(0)
        device = frames.device
        
        # CFG conditional dropout during training
        if self.training and torch.rand(1).item() < self.cfg_dropout_prob:
            # Unconditional forward pass - use zero context
            ctx = self.create_unconditional_context(B, device)
        else:
            # Conditional forward pass - use full context
            ctx = self.ctx_enc(frames, starts, goals, n_agents)
        
        # Use the SAME forward pass logic as your original model
        # Timestep sampling
        t = torch.randint(0, self.T_steps, (B,), device=device)
        
        # Forward diffusion with proper tensor handling
        x0 = trajs.permute(0, 1, 3, 2) # [B, Na, 2, T]
        x_noisy, noise = self.scheduler.q_sample(x0, t)
        x_noisy = x_noisy.permute(0, 1, 3, 2) # [B, Na, T, 2]
        
        # Agent masking - FIXED: Use max_agents consistently
        agent_mask = torch.arange(self.max_agents, device=device)[None, :] >= n_agents[:, None]
        
        # Process through network
        feat = self.preprocessor(x_noisy, agent_mask)
        temb = self.time_emb(t)
        cond = ctx + temb
        pred_noise = self.unet(feat, cond)
        
        # Use IDENTICAL loss computation as your original model
        return self._compute_enhanced_loss(pred_noise, noise, x_noisy, trajs, 
                                         starts, goals, n_agents, t, progress)
    
    def _compute_enhanced_loss(self, pred_noise, noise, x_noisy, trajs, 
                             starts, goals, n_agents, t, progress):
        """Identical loss computation to your original model"""
        B = pred_noise.size(0)
        device = pred_noise.device
        
        # Enhanced loss with Min-SNR weighting for better convergence
        snr = self.scheduler.alphas_cum[t] / (1 - self.scheduler.alphas_cum[t])
        min_snr_gamma = 5.0
        weight = torch.minimum(snr, torch.tensor(min_snr_gamma, device=device)) / snr
        
        # Apply weighting before masking
        agent_mask = torch.arange(self.max_agents, device=device)[None, :] >= n_agents[:, None]
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
        
        # Predicted x0 for additional losses
        alpha_t = self.scheduler.sqrt_ac[t].view(-1, 1, 1, 1)
        sigma_t = self.scheduler.sqrt_1mac[t].view(-1, 1, 1, 1)
        pred_x0 = (x_noisy - sigma_t * pred_noise) / (alpha_t + 1e-8)
        
        # Additional losses (identical to your original model)
        boundary_loss = self.compute_boundary_loss(pred_noise, noise.permute(0, 1, 3, 2), 
                                                 trajs, x_noisy, starts, goals, n_agents, t)
        velocity_loss = self.compute_velocity_constraints_loss(pred_x0, n_agents, max_velocity=2.0, dt=0.1)
        temporal_loss = self.compute_temporal_consistency_loss(pred_x0, n_agents)
        collision_loss = self.compute_collision_loss(pred_x0, n_agents)
        
        # Progressive loss weighting (identical to your original)
        if progress > 0.8:
            total_loss = 0.8*noise_pred_loss + 0.00*boundary_loss + 0.1*temporal_loss + 0.1*velocity_loss
        elif progress > 0.5:
            total_loss = 0.9*noise_pred_loss + 0.00*boundary_loss + 0.0*temporal_loss + 0.1*velocity_loss
        else:
            total_loss = 0.9*noise_pred_loss + 0.0*boundary_loss + 0.01*temporal_loss + 0.09*velocity_loss
        
        return 0.9*noise_pred_loss + 0.06*temporal_loss + 0.04*boundary_loss + 0.0*collision_loss
    
    @torch.no_grad()
    def sample_with_cfg(self, frames, starts, goals, n_agents, steps=50, 
                        guidance_scale=1.0, max_step_size=0.1):
        """CFG-enhanced sampling (identical to previous implementation)"""
        B = frames.size(0)
        device = frames.device
        
        # Initialize noise
        x = torch.randn(B, self.max_agents, self.horizon, 2, device=device)
        
        # Conditional and unconditional contexts
        ctx_conditional = self.ctx_enc(frames, starts, goals, n_agents)
        ctx_unconditional = self.create_unconditional_context(B, device)
        
        # Constraint setup (identical to your original model)
        constraint_mask = torch.zeros(B, self.max_agents, self.horizon, device=device, dtype=torch.bool)
        constraint_mask[:, :, 0] = True   # Fix start positions
        constraint_mask[:, :, -1] = True  # Fix goal positions
        
        constraint_values = torch.zeros(B, self.max_agents, self.horizon, 2, device=device)
        for b in range(B):
            n_active = n_agents[b].item() if isinstance(n_agents, torch.Tensor) else n_agents
            constraint_values[b, :n_active, 0] = starts[b, :n_active]
            constraint_values[b, :n_active, -1] = goals[b, :n_active]
        
        timesteps = torch.linspace(self.T_steps-1, 0, steps).long().to(device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            agent_mask = torch.arange(self.max_agents, device=device)[None, :] >= n_agents[:, None]
            
            # Get predictions from both conditional and unconditional models
            feat = self.preprocessor(x, agent_mask)
            temb = self.time_emb(t_batch)
            
            # Conditional prediction
            cond_conditional = ctx_conditional + temb
            eps_conditional = self.unet(feat, cond_conditional)
            
            # Unconditional prediction
            cond_unconditional = ctx_unconditional + temb
            eps_unconditional = self.unet(feat, cond_unconditional)
            
            # Apply classifier-free guidance
            if guidance_scale != 1.0:
                eps = eps_unconditional + guidance_scale * (eps_conditional - eps_unconditional)
            else:
                eps = eps_conditional
            
            # DDIM update (identical to your original)
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t = self.scheduler.alphas_cum[t]
                alpha_next = self.scheduler.alphas_cum[t_next]
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
                x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next) * eps
            else:
                alpha_t = self.scheduler.alphas_cum[t]
                x = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            
            # Apply constraints (identical to your original)
            x[constraint_mask.unsqueeze(-1).expand_as(x)] = constraint_values[constraint_mask.unsqueeze(-1).expand_as(constraint_values)]
        
        return x
    
    @torch.no_grad()
    def sample_full_trajectory_receding_horizon_cfg(self, frames, starts, goals, n_agents,
                                                full_horizon_length, horizon_size=8,
                                                guidance_scale=1.0, max_step_size=0.1):
        """Generate full trajectory using CFG-enhanced receding horizon sampling"""
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
            
            # Sample current horizon with CFG guidance
            horizon_prediction = self.sample_with_cfg(
                frames, current_positions, horizon_goals, n_agents,
                steps=50, guidance_scale=guidance_scale, max_step_size=max_step_size
            )
            
            # Store in full trajectory
            full_trajectory[:, :, start_t:end_t, :] = horizon_prediction
            
            # Update current positions for next horizon
            current_positions = horizon_prediction[:, :, -1, :].clone()
        
        return full_trajectory

