import os
import math
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import pdb

from MADP_diffusion_v2_1 import EnhancedMultiAgentDiffusionModel


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
        self.xy_std = all_xy_coords.std(dim=0)    # [2]
        self.xy_std = torch.clamp(self.xy_std, min=0.1)

        
        self.step_sizes_mean = all_step_sizes.mean(dim=0)  # [2]
        self.step_sizes_std = all_step_sizes.std(dim=0)    # [2]
        all_step_norm = (all_step_sizes - self.step_sizes_mean) / (3 * self.step_sizes_std)
        self.max_step = torch.max(all_step_norm)

    def __len__(self):
        """CRITICAL: This method must be implemented for DataLoader to work"""
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        frame = torch.tensor(self.h5['frames'][ep][0], dtype=torch.float32)
        traj_full = torch.tensor(self.h5['trajectories'][ep][:self.horizon], dtype=torch.float32)
        start_full = torch.tensor(self.h5['start_poses'][ep][:], dtype=torch.float32)
        goal_full = torch.tensor(self.h5['goal_poses'][ep][:], dtype=torch.float32)
        n_agents = traj_full.shape[1]
        
        # Extract and normalize only X-Y coordinates
        traj_xy = traj_full[..., :2]  # [T, Na, 2]
        start_xy = start_full[..., :2]  # [Na, 2]
        goal_xy = goal_full[..., :2]   # [Na, 2]
        
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
            'traj': traj_xy_norm,      # Now only [T, Na, 2]
            'start': start_xy_norm,    # Now only [Na, 2]
            'goal': goal_xy_norm,      # Now only [Na, 2]
            'n_agents': n_agents
        }
        # return {
        #     'frame': frame,
        #     'traj': traj_xy,      # Now only [T, Na, 2]
        #     'start': start_xy,    # Now only [Na, 2]
        #     'goal': goal_xy,      # Now only [Na, 2]
        #     'n_agents': n_agents
        # }


def collate_fn(batch):
    B = len(batch)
    T = batch[0]['traj'].shape[0]
    maxA = 10

    frames = torch.stack([b['frame'] for b in batch])
    starts = torch.zeros(B, maxA, 2)
    goals = torch.zeros(B, maxA, 2)
    trajs = torch.zeros(B, maxA, T, 2)
    n_agents = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        Ni = b['n_agents']
        n_agents[i] = Ni
        starts[i, :Ni] = b['start'][:Ni, :2]
        goals[i, :Ni] = b['goal'][:Ni, :2]
        trajs[i, :Ni] = b['traj'][:, :Ni, :2].permute(1, 0, 2)

    return frames, starts, goals, n_agents, trajs

def enhanced_curriculum_schedule(epoch, total_epochs, max_agents):
    """Enhanced curriculum with sigmoid progression"""
    progress = epoch / total_epochs
    # Sigmoid curriculum for smoother agent progression
    sigmoid_progress = 1 / (1 + math.exp(-10 * (progress - 0.5)))
    return max(1, int(max_agents * sigmoid_progress))

def train():
    # Configuration
    train_config_file = "MADP_training_config.yaml"
    try:
        with open(train_config_file, 'r') as file:
            train_config = yaml.safe_load(file)
        scenario_name = train_config['param']['scenario']
        num_agents = train_config['param']['num_agents']
        horizon = train_config['param']['horizon']
        h5_path = f"{scenario_name}_Na_{num_agents}_T_{horizon}_dataset.h5"
    except FileNotFoundError:
        # Default values if config file not found
        scenario_name = "default"
        num_agents = 4
        horizon = 20
        h5_path = "dataset.h5"  # Replace with your dataset path
    
    # Training parameters
    batch_size = 32
    epochs = 5000  # Reduced due to better convergence with enhanced scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training Enhanced MADP on {device}")
    print(f"Dataset: {h5_path}")
    print(f"Scenario: {scenario_name}, Agents: {num_agents}, Horizon: {horizon}")
    
    # Data loaders with normalized dataset
    train_ds = NormalizedTrajectoryDataset(h5_path, 'train', horizon=horizon)
    test_ds = NormalizedTrajectoryDataset(h5_path, 'test', horizon=horizon)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, 
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 1, shuffle=True, 
                           collate_fn=collate_fn, pin_memory=True)
    
    # Enhanced model with bounded noise scheduling
    model = EnhancedMultiAgentDiffusionModel(
        max_agents=10,
        horizon=horizon,
        state_dim=2,
        img_ch=3,
        hid=128,
        diffusion_steps=150,
        schedule_type='linear'  # Key improvement
    ).to(device)
    
    # Enhanced optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    writer = SummaryWriter(f"runs/enhanced_madp_{scenario_name}")
    
    # Create output directory
    os.makedirs("samples", exist_ok=True)
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Enhanced MADP"):
        model.train()
        current_agents = enhanced_curriculum_schedule(epoch, epochs, num_agents+1)
        total_loss = 0.0
        progress = epoch/epochs
        
        for frames, starts, goals, n_agents, trajs in train_loader:
            frames = frames.to(device, non_blocking=True)
            starts = starts.to(device, non_blocking=True)
            goals = goals.to(device, non_blocking=True)
            n_agents = n_agents.to(device, non_blocking=True)
            trajs = trajs.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            loss = model(frames, starts, goals, n_agents, trajs, current_agents, progress)
            loss.backward()
            
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        lr_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        # Logging
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        writer.add_scalar("Curriculum/Active_Agents", current_agents, epoch)
        writer.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        
        # Periodic evaluation and visualization
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}, Active Agents = {current_agents}")
            model.eval()
            with torch.no_grad():
                # Sample evaluation
                frame, start, goal, na, tr = next(iter(test_loader))
                frame, start, goal, na, tr = frame.to(device), start.to(device), goal.to(device), na.to(device), tr.to(device)
                xy_mean = train_ds.xy_mean.to(tr.device)
                xy_std = train_ds.xy_std.to(tr.device)
                tr_denorm = tr * (3 * xy_std) + xy_mean
                
                # max_step = train_ds.max_step
                max_step = 0.2
                max_step_size_norm = (max_step - xy_mean) / (3 * xy_std)
                max_step_size_norm = torch.norm(max_step_size_norm)
                # pdb.set_trace()
                
                # Fast sampling with DDIM
                # out = model.sample(frames=f, starts=s, goals=g, n_agents=n, steps=50)
                # out = model.sample(frames=frame, starts=start, goals=goal, n_agents=na, steps=250, max_step_size=max_step_size_norm)
                out = model.sample_with_constraints(frames=frame, starts=start, goals=goal, n_agents=na, steps=250, max_step_size=max_step_size_norm, dataset=train_ds)
                xy_mean = train_ds.xy_mean.to(out.device)
                xy_std = train_ds.xy_std.to(out.device)
                out_denorm = out * (3 * xy_std) + xy_mean
                
                
                # Enhanced visualization
                fig, ax = plt.subplots(figsize=(12, 10))
                colors = ['blue', 'red', 'green', 'orange', 'purple']
                
                for i in range(min(na.item(), 4)):
                    gt = tr_denorm[0, i].cpu().numpy()
                    pr = out_denorm[0, i].cpu().numpy()
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
                ax.set_title(f'Enhanced MADP - Epoch {epoch}\nBounded Cosine Scheduler', fontsize=16)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                plt.tight_layout()
                plt.savefig(f"samples/enhanced_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
                plt.close()
                
    
    # Save final model
    writer.close()
    torch.save(model.state_dict(), f"enhanced_madp_{scenario_name}.pth")
    print(f"Enhanced training completed! Model saved as enhanced_madp_{scenario_name}.pth")

if __name__ == "__main__":
    train()
