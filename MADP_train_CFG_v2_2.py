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

from MADP_diffusion_CFG_v2_2 import CFGEnhancedMultiAgentDiffusionModel
from MADP_train_and_sample_v2_2 import (NormalizedTrajectoryDataset, collate_fn, 
                                       enhanced_curriculum_schedule)

def process_receding_horizon_training_cfg(model, optimizer, frames, starts, goals, n_agents, full_trajs,
                                        horizon_size=10, progress=0.0, device='cuda'):
    """CFG-enhanced receding horizon processing - IDENTICAL to your original approach"""
    B, Na, T_full, D = full_trajs.shape
    total_batch_loss = 0.0
    current_positions = starts.clone()

    # Simple calculation - no edge case handling needed
    num_horizons = T_full // horizon_size # Always exact division

    for horizon_idx in range(num_horizons):
        start_t = horizon_idx * horizon_size
        end_t = start_t + horizon_size # Always exactly horizon_size

        # Extract fixed-size horizon trajectory
        horizon_trajs = full_trajs[:, :, start_t:end_t, :] # Always [B, Na, horizon_size, 2]

        # Set current positions as start for this horizon
        horizon_trajs[:, :, 0, :] = current_positions

        # Set intermediate goals for non-final horizons (IDENTICAL to your approach)
        if horizon_idx == num_horizons - 1:
            # Final horizon - use actual goals
            horizon_goals = goals
        else:
            # Intermediate horizon - interpolate toward final goals
            progress_to_goal = (horizon_idx + 1) / num_horizons
            horizon_goals = current_positions + progress_to_goal * (goals - current_positions)

        # Standard training step with CFG
        optimizer.zero_grad()
        
        # Forward pass through CFG model (will randomly apply conditional dropout)
        diffusion_loss = model(frames, current_positions, horizon_goals, n_agents,
                             horizon_trajs, n_agents, progress)

        # Add goal-reaching loss for current horizon (IDENTICAL to your approach)
        with torch.no_grad():
            # Quick prediction to compute goal loss
            alpha_t = model.scheduler.alphas_cum[torch.randint(0, model.T_steps, (1,), device=device)]
            pred_x0_approx = horizon_trajs # Approximation for goal loss

            goal_reaching_loss = model.compute_receding_horizon_goal_loss(
                pred_x0_approx, horizon_goals, n_agents
            )

        # Combined loss (IDENTICAL to your approach)
        goal_weight = 0.1
        total_horizon_loss = diffusion_loss + goal_weight * goal_reaching_loss

        # Backward pass
        total_horizon_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_batch_loss += total_horizon_loss.item()

        # Update positions for next horizon (IDENTICAL to your approach)
        with torch.no_grad():
            current_positions = full_trajs[:, :, end_t-1, :].clone()

    return total_batch_loss / num_horizons

def train_cfg_with_receding_horizon():
    """Complete CFG training with receding horizon approach"""
    
    # Load configuration (IDENTICAL to your approach)
    train_config_file = "MADP_training_config.yaml"
    try:
        with open(train_config_file, 'r') as file:
            train_config = yaml.safe_load(file)
        scenario_name = train_config['param']['scenario']
        num_agents = train_config['param']['num_agents']
        full_horizon = train_config['param']['horizon']
        h5_path = f"{scenario_name}_Na_{num_agents}_T_{full_horizon}_dataset.h5"
        diffusion_steps = train_config['param']['diffuse_steps']
        max_epochs = train_config['param']['n_epochs']
    except FileNotFoundError:
        scenario_name = "default"
        num_agents = 4
        full_horizon = 40
        h5_path = "dataset.h5"
        diffusion_steps = 150  # Match your existing checkpoints
        max_epochs = 1000

    # Training parameters (IDENTICAL to your approach)
    batch_size = 16
    epochs = max_epochs
    horizon_size = 8  # Fixed horizon size for receding horizon
    cfg_dropout_prob = 0.1  # 10% unconditional training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training CFG-Enhanced MADP with Receding Horizon on {device}")
    print(f"Dataset: {h5_path}")
    print(f"CFG Dropout Probability: {cfg_dropout_prob}")
    print(f"Scenario: {scenario_name}, Agents: {num_agents}")
    print(f"Full Horizon: {full_horizon}, Training Horizon: {horizon_size}")

    # Data loaders (IDENTICAL to your approach)
    train_ds = NormalizedTrajectoryDataset(h5_path, 'train', horizon=full_horizon)
    test_ds = NormalizedTrajectoryDataset(h5_path, 'test', horizon=full_horizon)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 1, shuffle=True,
                            collate_fn=collate_fn, pin_memory=True)

    # CFG-Enhanced Model with IDENTICAL architecture
    model = CFGEnhancedMultiAgentDiffusionModel(
        max_agents=10,
        horizon=horizon_size,  # Use fixed horizon size for training
        state_dim=2,
        img_ch=3,
        hid=128,
        diffusion_steps=diffusion_steps,
        schedule_type='linear',
        cfg_dropout_prob=cfg_dropout_prob
    ).to(device)

    print(f"CFG Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer and scheduler (IDENTICAL to your approach)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Logging
    writer = SummaryWriter(f"runs/cfg_horizon_madp_{scenario_name}")
    os.makedirs("samples_cfg_horizon", exist_ok=True)

    # Training loop (IDENTICAL structure to your approach)
    print("Starting CFG training with receding horizon...")
    for epoch in tqdm(range(epochs), desc="CFG Receding Horizon Training"):
        model.train()
        current_agents = enhanced_curriculum_schedule(epoch, epochs, num_agents+1)
        total_loss = 0.0
        progress = epoch / epochs

        for frames, starts, goals, n_agents, full_trajs in train_loader:
            frames = frames.to(device, non_blocking=True)
            starts = starts.to(device, non_blocking=True)
            goals = goals.to(device, non_blocking=True)
            n_agents = n_agents.to(device, non_blocking=True)
            full_trajs = full_trajs.to(device, non_blocking=True)

            # Process each trajectory with CFG receding horizon
            batch_loss = process_receding_horizon_training_cfg(
                model, optimizer, frames, starts, goals, n_agents, full_trajs,
                horizon_size=horizon_size, progress=progress, device=device
            )

            total_loss += batch_loss

        avg_loss = total_loss / len(train_loader)
        lr_scheduler.step()

        # Logging
        writer.add_scalar("Loss/Train_CFG_Horizon", avg_loss, epoch)
        writer.add_scalar("Curriculum/Active_Agents", current_agents, epoch)
        writer.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Training/Progress", progress, epoch)
        writer.add_scalar("Training/CFG_Dropout_Prob", cfg_dropout_prob, epoch)

        # Periodic evaluation and visualization (IDENTICAL to your approach)
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: CFG Horizon Loss = {avg_loss:.6f}, Active Agents = {current_agents}, Progress = {progress:.3f}")
            
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

                # Generate predictions with different guidance scales
                guidance_scales = [1.0, 2.0, 3.0]
                predictions = {}
                
                for scale in guidance_scales:
                    pred = model.sample_with_cfg(
                        frame, start, goal, na, steps=50, guidance_scale=scale
                    )
                    predictions[scale] = pred * (3 * xy_std) + xy_mean

                # Enhanced visualization (IDENTICAL structure to your approach)
                fig, axes = plt.subplots(1, len(guidance_scales) + 1, figsize=(20, 5))
                colors = ['blue', 'red', 'green', 'orange', 'purple']
                
                # Ground truth
                ax = axes[0]
                for i in range(min(na.item(), 4)):
                    gt = full_traj_denorm[0, i].cpu().numpy()
                    color = colors[i % len(colors)]
                    ax.plot(gt[:, 0], gt[:, 1], '--', color=color,
                           label=f'GT Agent {i}', alpha=0.8, linewidth=3)
                    ax.scatter(gt[0, 0], gt[0, 1], c='green', s=100, marker='v', edgecolors='black')
                    ax.scatter(gt[-1, 0], gt[-1, 1], c='red', s=100, marker='s', edgecolors='black')
                
                ax.set_title('Ground Truth', fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')

                # CFG predictions
                for idx, scale in enumerate(guidance_scales):
                    ax = axes[idx + 1]
                    pred = predictions[scale]
                    
                    for i in range(min(na.item(), 4)):
                        pr = pred[0, i].cpu().numpy()
                        color = colors[i % len(colors)]
                        ax.plot(pr[:, 0], pr[:, 1], '-o', color=color,
                               label=f'CFG Agent {i}', alpha=0.8, linewidth=2)
                        ax.scatter(pr[0, 0], pr[0, 1], c='green', s=100, marker='v', edgecolors='black')
                        ax.scatter(pr[-1, 0], pr[-1, 1], c='red', s=100, marker='^', edgecolors='black')
                    
                    ax.set_title(f'CFG w={scale}', fontsize=14)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_aspect('equal')

                plt.suptitle(f'CFG Receding Horizon MADP - Epoch {epoch}', fontsize=16)
                plt.tight_layout()
                plt.savefig(f"samples_cfg_horizon/cfg_horizon_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
                plt.close()

    # Save final model
    writer.close()
    torch.save(model.state_dict(), f"cfg_horizon_madp_{scenario_name}.pth")
    
    # Also save model configuration for later loading
    config_to_save = {
        'max_agents': 10,
        'horizon': horizon_size,
        'state_dim': 2,
        'img_ch': 3,
        'hid': 128,
        'diffusion_steps': diffusion_steps,
        'schedule_type': 'linear',
        'cfg_dropout_prob': cfg_dropout_prob,
        'scenario_name': scenario_name,
        'num_agents': num_agents,
        'full_horizon': full_horizon
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config_to_save
    }, f"cfg_horizon_madp_{scenario_name}_with_config.pth")
    
    print(f"CFG receding horizon training completed!")
    print(f"Models saved:")
    print(f"  - cfg_horizon_madp_{scenario_name}.pth")
    print(f"  - cfg_horizon_madp_{scenario_name}_with_config.pth")

if __name__ == "__main__":
    train_cfg_with_receding_horizon()
