import os
import math
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import pdb

from MADP_diffusion_v2 import MultiAgentDiffusionModel

class TrajectoryDataset(Dataset):
    def __init__(self, h5_path, split='train', horizon=20,
                 train_ratio=0.8, max_agents=10):
        self.h5 = h5py.File(h5_path, 'r')
        all_eps = sorted(self.h5['frames'].keys())
        n = len(all_eps)
        cut = int(n * train_ratio)
        self.episodes = all_eps[:cut] if split=='train' else all_eps[cut:]
        self.horizon = horizon
        self.max_agents = max_agents

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        frame = torch.tensor(self.h5['frames'][ep][0],
                           dtype=torch.float32)
        traj = torch.tensor(self.h5['trajectories'][ep][:self.horizon],
                          dtype=torch.float32)
        start = torch.tensor(self.h5['start_poses'][ep][:],
                           dtype=torch.float32)
        goal = torch.tensor(self.h5['goal_poses'][ep][:],
                          dtype=torch.float32)
        n_agents = traj.shape[1]
        return {
            'frame': frame,
            'traj': traj,
            'start': start,
            'goal': goal,
            'n_agents': n_agents
        }

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
        trajs[i, :Ni] = b['traj'][:, :Ni, :2].permute(1,0,2)   # Currently storing only actions.

    return frames, starts, goals, n_agents, trajs

def train():
    # Clear any existing tqdm instances to prevent multiline issues
    tqdm._instances.clear()
    
    # Load configuration
    train_config_file = "MADP_training_config.yaml"
    with open(train_config_file, 'r') as file:
        train_config = yaml.safe_load(file)
    
    scenario_name = train_config['param']['scenario']
    num_agents = train_config['param']['num_agents']
    horizon = train_config['param']['horizon']

    # Setup
    h5_path = f"{scenario_name}_Na_{num_agents}_T_{horizon}_dataset.h5"
    batch_size = 64
    epochs = 2000
    eval_interval = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loaders
    train_ds = TrajectoryDataset(h5_path, 'train')
    test_ds = TrajectoryDataset(h5_path, 'test')
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                            collate_fn=collate_fn,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 1, shuffle=False,
                           collate_fn=collate_fn,
                           pin_memory=True)

    # Model and optimizer
    model = MultiAgentDiffusionModel(
        max_agents=10,
        horizon=horizon,
        state_dim=2,
        img_ch=3,
        hid=128,
        diffusion_steps=500,
        t_min=50
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # TensorBoard writer with proper run naming
    writer = SummaryWriter(f"runs/madp_curriculum_{scenario_name}")

    # Curriculum learning schedule
    start_agents = 1
    end_agents = 4
    step = math.ceil(epochs / (end_agents - start_agents + 1))
    schedule = {i*step: start_agents + i
                for i in range(end_agents - start_agents + 1)}

    # Loss tracking for plotting
    train_losses = []
    eval_losses = []
    
    # Training loop with proper tqdm configuration
    current_agents = start_agents
    
    # Main epoch progress bar
    epoch_pbar = tqdm(range(epochs), desc="Training Progress", 
                      position=0, leave=True, ascii=True)
    
    for ep in epoch_pbar:
        model.train()
        current_agents = schedule.get(ep, current_agents)
        total_loss = 0.0
        num_batches = 0

        # Batch progress bar with fixed position
        batch_pbar = tqdm(train_loader, 
                         desc=f"Epoch {ep+1}/{epochs} (Agents: {current_agents})",
                         position=1, leave=False, ascii=True)
        
        for frames, starts, goals, n_agents, trajs in batch_pbar:
            # Move to device
            frames = frames.to(device, non_blocking=True)
            starts = starts.to(device, non_blocking=True)
            goals = goals.to(device, non_blocking=True)
            n_agents = n_agents.to(device, non_blocking=True)
            trajs = trajs.to(device, non_blocking=True)

            # Forward pass
            optimizer.zero_grad()
            loss = model(frames, starts, goals, n_agents,
                        trajs, current_agents)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update batch progress bar with current loss
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/num_batches:.4f}'
            })

        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Log to TensorBoard
        writer.add_scalar("Loss/Train", avg_loss, ep)
        writer.add_scalar("Curriculum/Active_Agents", current_agents, ep)
        writer.add_scalar("Training/Learning_Rate", 
                         optimizer.param_groups[0]['lr'], ep)

        # Update main progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{avg_loss:.4f}',
            'Agents': current_agents
        })

        # Periodic evaluation and sampling
        if ep % eval_interval == 0 or ep == epochs - 1:
            model.eval()
            eval_loss = 0.0
            eval_batches = 0
            
            with torch.no_grad():
                # Evaluation loop
                eval_pbar = tqdm(test_loader, desc="Evaluating", 
                               position=2, leave=False, ascii=True)
                
                for frames, starts, goals, n_agents, trajs in eval_pbar:
                    frames = frames.to(device, non_blocking=True)
                    starts = starts.to(device, non_blocking=True)
                    goals = goals.to(device, non_blocking=True)
                    n_agents = n_agents.to(device, non_blocking=True)
                    trajs = trajs.to(device, non_blocking=True)
                    
                    loss = model(frames, starts, goals, n_agents,
                               trajs, current_agents)
                    eval_loss += loss.item()
                    eval_batches += 1
                
                avg_eval_loss = eval_loss / len(test_loader)
                eval_losses.append(avg_eval_loss)
                
                # Log evaluation metrics
                writer.add_scalar("Loss/Validation", avg_eval_loss, ep)
                
                # Sample generation for visualization
                f, s, g, n, tr = next(iter(test_loader))
                f = f.to(device, non_blocking=True)
                s = s.to(device, non_blocking=True)
                g = g.to(device, non_blocking=True)
                n = n.to(device, non_blocking=True)
                
                out = model.sample(frames=f, starts=s, goals=g, n_agents=n)

                # Create and save trajectory plot
                fig, ax = plt.subplots(figsize=(10, 8))
                for i in range(min(n.item(), 5)):  # Limit to 5 agents for clarity
                    gt = tr[0, i].cpu().numpy()
                    pr = out[0, i].cpu().numpy()
                    ax.plot(gt[:, 0], gt[:, 1], '--', 
                           label=f'GT Agent {i}' if i < 2 else "", 
                           alpha=0.7, linewidth=2)
                    ax.plot(pr[:, 0], pr[:, 1], '-', 
                           label=f'Pred Agent {i}' if i < 2 else "", 
                           alpha=0.7, linewidth=2)
                    # Mark start and end points
                    ax.scatter(gt[0, 0], gt[0, 1], c='green', s=50, marker='o')
                    ax.scatter(gt[-1, 0], gt[-1, 1], c='red', s=50, marker='s')
                
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.set_title(f'Trajectory Comparison - Epoch {ep}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"samples/epoch_{ep}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)

                # Create and save loss plot
                if len(train_losses) > 1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    epochs_range = range(len(train_losses))
                    ax.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
                    
                    if eval_losses:
                        eval_epochs = [i * eval_interval for i in range(len(eval_losses))]
                        ax.plot(eval_epochs, eval_losses, 'r-', label='Validation Loss', linewidth=2)
                    
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title('Training and Validation Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')  # Log scale for better visualization
                    plt.tight_layout()
                    plt.savefig(f"samples/loss_plot_epoch_{ep}.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)

                # Log images to TensorBoard
                writer.add_figure("Trajectories/Comparison", fig, ep)
                
                tqdm.write(f"Epoch {ep}: Train Loss = {avg_loss:.4f}, "
                          f"Val Loss = {avg_eval_loss:.4f}")

    # Final cleanup and save
    epoch_pbar.close()
    writer.close()
    torch.save(model.state_dict(), f"madp_diffusion_{scenario_name}.pth")
    
    # Save final loss plot
    fig, ax = plt.subplots(figsize=(12, 8))
    epochs_range = range(len(train_losses))
    ax.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if eval_losses:
        eval_epochs = [i * eval_interval for i in range(len(eval_losses))]
        ax.plot(eval_epochs, eval_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Final Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig("samples/final_loss_plot.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Training completed! Final model saved as madp_diffusion_{scenario_name}.pth")

if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)
    train()
