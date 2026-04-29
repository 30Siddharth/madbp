import os
import math
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import pdb

from MADP_diffusion_v2_2 import EnhancedMultiAgentDiffusionModel

class NormalizedTrajectoryDataset(Dataset):
    def __init__(self, h5_path, split='train', horizon=20, train_ratio=0.8, max_agents=10):
        self.h5 = h5py.File(h5_path, 'r')
        all_eps = sorted(self.h5['frames'].keys())
        n = len(all_eps)
        cut = int(n * train_ratio)
        self.episodes = all_eps[:cut] if split == 'train' else all_eps[cut:]
        self.horizon = horizon
        self.max_agents = max_agents
        
        # Compute normalization stats for X-Y only
        self._compute_xy_normalization_stats()
    
    def _compute_xy_normalization_stats(self):
        """Compute normalization statistics for X-Y coordinates only"""
        all_xy_coords = []
        all_step_sizes = []
        
        for ep in self.episodes[:min(100, len(self.episodes))]:
            traj = torch.tensor(self.h5['trajectories'][ep][:self.horizon])
            # Extract only X-Y coordinates (first 2 dimensions)
            xy_coords = traj[..., :2]  # Shape: [T, Na, 2]
            all_xy_coords.append(xy_coords.reshape(-1, 2))
            
            # Compute step sizes between consecutive points
            velocities = xy_coords[1:] - xy_coords[:-1]  # [T-1, Na, 2]
            step_sizes = torch.norm(velocities, dim=-1)  # [T-1, Na]
            all_step_sizes.append(step_sizes.reshape(-1))
        
        all_xy_coords = torch.cat(all_xy_coords, dim=0)
        all_step_sizes = torch.cat(all_step_sizes, dim=0)
        
        self.xy_mean = all_xy_coords.mean(dim=0)  # [2]
        self.xy_std = all_xy_coords.std(dim=0)  # [2]
        self.xy_std = torch.clamp(self.xy_std, min=0.1)
        
        self.step_sizes_mean = all_step_sizes.mean(dim=0)
        self.step_sizes_std = all_step_sizes.std(dim=0)
        all_step_norm = (all_step_sizes - self.step_sizes_mean) / (3 * self.step_sizes_std)
        self.max_step = torch.max(all_step_norm)
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        frame = torch.tensor(self.h5['frames'][ep][0], dtype=torch.float32)
        traj_full = torch.tensor(self.h5['trajectories'][ep][:], dtype=torch.float32)  # Full trajectory
        start_full = torch.tensor(self.h5['start_poses'][ep][:], dtype=torch.float32)
        goal_full = torch.tensor(self.h5['goal_poses'][ep][:], dtype=torch.float32)
        
        n_agents = traj_full.shape[1]
        
        # Extract and normalize only X-Y coordinates
        traj_xy = traj_full[..., :2]  # [T_full, Na, 2]
        start_xy = start_full[..., :2]  # [Na, 2]
        goal_xy = goal_full[..., :2]  # [Na, 2]
        
        # Normalize X-Y coordinates
        traj_xy_norm = (traj_xy - self.xy_mean) / (3 * self.xy_std)
        start_xy_norm = (start_xy - self.xy_mean) / (3 * self.xy_std)
        goal_xy_norm = (goal_xy - self.xy_mean) / (3 * self.xy_std)
        
        # Clamp to [-1, 1] range
        traj_xy_norm = torch.clamp(traj_xy_norm, -1, 1)
        start_xy_norm = torch.clamp(start_xy_norm, -1, 1)
        goal_xy_norm = torch.clamp(goal_xy_norm, -1, 1)
        
        return {
            'frame': frame,
            'traj': traj_xy_norm,  # Full trajectory [T_full, Na, 2]
            'start': start_xy_norm,  # [Na, 2]
            'goal': goal_xy_norm,  # [Na, 2]
            'n_agents': n_agents
        }

def collate_fn(batch):
    B = len(batch)
    T_full = batch[0]['traj'].shape[0]  # Full trajectory length
    maxA = 10
    
    frames = torch.stack([b['frame'] for b in batch])
    starts = torch.zeros(B, maxA, 2)
    goals = torch.zeros(B, maxA, 2)
    trajs = torch.zeros(B, maxA, T_full, 2)
    n_agents = torch.zeros(B, dtype=torch.long)
    
    for i, b in enumerate(batch):
        Ni = b['n_agents']
        n_agents[i] = Ni
        starts[i, :Ni] = b['start'][:Ni, :2]
        goals[i, :Ni] = b['goal'][:Ni, :2]
        trajs[i, :Ni] = b['traj'][:, :Ni, :2].permute(1, 0, 2)
    
    return frames, starts, goals, n_agents, trajs

def process_receding_horizon_training_simplified(model, optimizer, frames, starts, goals, n_agents, full_trajs, 
                                               horizon_size=10, progress=0.0, device='cuda'):
    """Simplified receding horizon processing with boundary constraints"""
    B, Na, T_full, D = full_trajs.shape
    
    total_batch_loss = 0.0
    current_positions = starts.clone()
    
    # Simple calculation - no edge case handling needed
    num_horizons = T_full // horizon_size  # Always exact division
    
    for horizon_idx in range(num_horizons):
        start_t = horizon_idx * horizon_size
        end_t = start_t + horizon_size  # Always exactly horizon_size
        
        # Extract fixed-size horizon trajectory
        horizon_trajs = full_trajs[:, :, start_t:end_t, :]  # Always [B, Na, 10, 2]
        
        # Set current positions as start for this horizon
        horizon_trajs[:, :, 0, :] = current_positions
        
        # Standard training step
        optimizer.zero_grad()
        
        # Forward pass through model
        pdb.set_trace()
        diffusion_loss = model(frames, current_positions, goals, n_agents, 
                              horizon_trajs, n_agents.max().item(), progress)
        
        # Add goal-reaching loss for current horizon
        with torch.no_grad():
            # Quick prediction to compute goal loss
            alpha_t = model.scheduler.alphas_cum[torch.randint(0, model.T_steps, (1,), device=device)]
            pred_x0_approx = horizon_trajs  # Approximation for goal loss
            
        goal_reaching_loss = model.compute_receding_horizon_goal_loss(
            pred_x0_approx, goals, n_agents
        )
        
        # Combined loss
        goal_weight = 0.1 if progress < 0.5 else 0.2
        total_horizon_loss = diffusion_loss + goal_weight * goal_reaching_loss
        
        # Backward pass
        total_horizon_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_batch_loss += total_horizon_loss.item()
        
        # Update positions for next horizon - always use last position from current horizon
        with torch.no_grad():
            current_positions = full_trajs[:, :, end_t-1, :].clone()
    
    return total_batch_loss / num_horizons

def train_with_receding_horizon():
    # Configuration
    train_config_file = "MADP_training_config.yaml"
    try:
        with open(train_config_file, 'r') as file:
            train_config = yaml.safe_load(file)
        scenario_name = train_config['param']['scenario']
        num_agents = train_config['param']['num_agents']
        full_horizon = train_config['param']['horizon']  # Full trajectory length
        h5_path = f"{scenario_name}_Na_{num_agents}_T_{full_horizon}_dataset.h5"
    except FileNotFoundError:
        # Default values if config file not found
        scenario_name = "default"
        num_agents = 4
        full_horizon = 40  # Full trajectory length
        h5_path = "dataset.h5"
    
    # Training parameters
    batch_size = 16  # Reduced due to more frequent updates
    epochs = 3000
    horizon_size = 8  # Fixed horizon size for receding horizon
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training Receding Horizon MADP with Boundary Constraints on {device}")
    print(f"Dataset: {h5_path}")
    print(f"Scenario: {scenario_name}, Agents: {num_agents}")
    print(f"Full Horizon: {full_horizon}, Training Horizon: {horizon_size}")
    
    # Data loaders
    train_ds = NormalizedTrajectoryDataset(h5_path, 'train', horizon=full_horizon)
    test_ds = NormalizedTrajectoryDataset(h5_path, 'test', horizon=full_horizon)
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 1, shuffle=True,
                            collate_fn=collate_fn, pin_memory=True)
    
    # Model with fixed horizon size for training
    model = EnhancedMultiAgentDiffusionModel(
        max_agents=10,
        horizon=horizon_size,  # Use fixed horizon size
        state_dim=2,
        img_ch=3,
        hid=128,
        diffusion_steps=250,
        schedule_type='linear'
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    writer = SummaryWriter(f"runs/temporal_madp_{scenario_name}")
    os.makedirs("samples", exist_ok=True)
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Boundary-Constrained Receding Horizon Training"):
        model.train()
        total_loss = 0.0
        progress = epoch / epochs
        
        for frames, starts, goals, n_agents, full_trajs in train_loader:
            frames = frames.to(device, non_blocking=True)
            starts = starts.to(device, non_blocking=True)
            goals = goals.to(device, non_blocking=True)
            n_agents = n_agents.to(device, non_blocking=True)
            full_trajs = full_trajs.to(device, non_blocking=True)
            pdb.set_trace()
            # Process each trajectory with receding horizon
            batch_loss = process_receding_horizon_training_simplified(
                model, optimizer, frames, starts, goals, n_agents, full_trajs,
                horizon_size=horizon_size, progress=progress, device=device
            )
            
            total_loss += batch_loss
        
        avg_loss = total_loss / len(train_loader)
        lr_scheduler.step()
        
        # Logging
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        writer.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Training/Progress", progress, epoch)
        
        # Periodic evaluation and visualization
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}, Progress = {progress:.3f}")
            
            model.eval()
            with torch.no_grad():
                # Sample evaluation
                frame, start, goal, na, full_traj = next(iter(test_loader))
                frame = frame.to(device)
                start = start.to(device)
                goal = goal.to(device)
                na = na.to(device)
                full_traj = full_traj.to(device)
                
                # Denormalization parameters
                xy_mean = train_ds.xy_mean.to(device)
                xy_std = train_ds.xy_std.to(device)
                
                # Denormalize ground truth
                full_traj_denorm = full_traj * (3 * xy_std) + xy_mean
                
                # Generate full trajectory using receding horizon sampling with boundary constraints
                predicted_full_traj = model.sample_full_trajectory_receding_horizon(
                    frame, start, goal, na, 
                    full_horizon_length=full_traj.size(2),
                    horizon_size=horizon_size,
                    max_step_size=0.1
                )
                
                # Denormalize predictions
                predicted_full_traj_denorm = predicted_full_traj * (3 * xy_std) + xy_mean
                
                # Enhanced visualization
                fig, ax = plt.subplots(figsize=(12, 10))
                colors = ['blue', 'red', 'green', 'orange', 'purple']
                
                for i in range(min(na.item(), 4)):
                    gt = full_traj_denorm[0, i].cpu().numpy()
                    pr = predicted_full_traj_denorm[0, i].cpu().numpy()
                    color = colors[i % len(colors)]
                    
                    ax.plot(gt[:, 0], gt[:, 1], '--', color=color,
                           label=f'GT Agent {i}', alpha=0.8, linewidth=3)
                    ax.plot(pr[:, 0], pr[:, 1], '-', color=color,
                           label=f'Pred Agent {i}', alpha=0.8, linewidth=3)
                    
                    # Mark start (green circle) and goal (red square)
                    ax.scatter(gt[0, 0], gt[0, 1], c='green', s=100, marker='o',
                              edgecolors='black', linewidth=2)
                    ax.scatter(gt[-1, 0], gt[-1, 1], c='red', s=100, marker='s',
                              edgecolors='black', linewidth=2)
                
                ax.set_xlabel('X Position', fontsize=14)
                ax.set_ylabel('Y Position', fontsize=14)
                ax.set_title(f'Boundary-Constrained Receding Horizon MADP - Epoch {epoch}\nHorizon Size: {horizon_size}', fontsize=16)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
                plt.tight_layout()
                plt.savefig(f"samples/boundary_constrained_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
                plt.close()
    
    # Save final model
    writer.close()
    torch.save(model.state_dict(), f"boundary_constrained_madp_{scenario_name}.pth")
    print(f"Boundary-constrained receding horizon training completed! Model saved as boundary_constrained_madp_{scenario_name}.pth")

if __name__ == "__main__":
    train_with_receding_horizon()
