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

from MADP_intra_attention import MultiAgentDiffusionPolicy

# ===== Dataset and Training Functions =====

class TrajectoryDataset(Dataset):
    def __init__(self, h5_path, num_samples=None, split='train', horizon=10, train_ratio=0.8):
        self.h5_file = h5py.File(h5_path, 'r')
        total_samples = len(self.h5_file['frames'])
        
        if num_samples is None:
            num_samples = total_samples
            
        # Create train/test indices
        all_indices = np.arange(total_samples)
        np.random.shuffle(all_indices)
        split_idx = int(train_ratio * total_samples)
        
        if split == 'train':
            self.indices = all_indices[:split_idx]
        else:  # test
            self.indices = all_indices[split_idx:]
            
        self.num_samples = len(self.indices)
        self.horizon = horizon
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Map idx to the actual episode index
        actual_idx = self.indices[idx]
        episode_name = f'episode_{actual_idx}'
        
        # Load data from HDF5 file
        frames = torch.tensor(self.h5_file['frames'][episode_name][0:1], dtype=torch.float32)  # Only take first frame
        states_actions = torch.tensor(self.h5_file['trajectories'][episode_name][:self.horizon], dtype=torch.float32)
        start_poses = torch.tensor(self.h5_file['start_poses'][episode_name][:], dtype=torch.float32)
        goal_poses = torch.tensor(self.h5_file['goal_poses'][episode_name][:], dtype=torch.float32)
        num_agents = self.h5_file.attrs['num_agents']
        
        # Create context tuple properly
        context = (frames, start_poses, goal_poses, torch.tensor([num_agents]))
        
        return {
            'states_actions': states_actions,
            'context': context
        }

def train_with_curriculum(model, dataloader, optimizer, num_epochs, device,
                         start_agents=1, max_agents=4, curriculum_epochs=None):
    """
    Train the model with curriculum learning by gradually increasing the number of agents
    Args:
        model: The diffusion model
        dataloader: Data loader for training data
        optimizer: Optimizer for training
        num_epochs: Total number of training epochs
        device: Device to train on
        start_agents: Number of agents to start training with
        max_agents: Maximum number of agents
        curriculum_epochs: List of epoch numbers at which to increase the number of agents
                          If None, will increase linearly throughout training
    """
    if curriculum_epochs is None:
        # Create a linear curriculum
        curriculum_epochs = [int(num_epochs * i / (max_agents - start_agents + 1))
                            for i in range(1, max_agents - start_agents + 1)]
    
    current_agents = start_agents
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        batches = 0
        
        # Check if we need to increase the number of agents
        if epoch in curriculum_epochs and current_agents < max_agents:
            current_agents += 1
            print(f"Epoch {epoch}: Increasing to {current_agents} agents")
            
        for batch in dataloader:
            # Unpack batch data
            trajectories, context = batch
            trajectories = trajectories.to(device)
            context = [c.to(device) for c in context]
            
            # Set active agents for this batch
            batch_size = trajectories.shape[0]
            active_agents = torch.full((batch_size,), current_agents, device=device, dtype=torch.long)
            
            # Forward pass
            optimizer.zero_grad()
            loss = model(trajectories, context, active_agents)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        avg_loss = total_loss / batches
        print(f"Epoch {epoch}, Agents: {current_agents}, Avg Loss: {avg_loss:.4f}")
        
    return model

def train_diffusion_model(model, dataloader, test_dataloader=None, optimizer=None,
                         num_epochs=100, eval_interval=10, device="cuda"):
    model.to(device)
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    best_test_loss = float('inf')
    pbar = tqdm(total=num_epochs, desc="epoch loss = 0")
    writer = SummaryWriter(log_dir='runs/madp_training')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            states_actions = batch['states_actions'].to(device)
            context = tuple(c.to(device) for c in batch['context'])
            
            optimizer.zero_grad()
            loss = model(states_actions, context)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Every few epochs, generate and log a sample trajectory
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                test_batch = next(iter(test_dataloader))
                test_states_actions = test_batch['states_actions'].to(device)
                test_context = tuple(c.to(device) for c in test_batch['context'])
                
                # Get number of agents from context
                num_agents = test_context[3].item()
                
                generated_trajectory = model.sample_trajectory(test_context, num_agents, save_steps=False)
                
                # Create a figure with trajectories
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
                
                # Plot ground truth positions
                pos_des = torch.flatten(test_states_actions, end_dim=1).cpu()
                for agent_id in range(num_agents):
                    des_xy = pos_des[:, agent_id]
                    ax1.plot(des_xy[:, 0], des_xy[:, 1], color=colors[agent_id],
                            linestyle='--', label=f'Agent {agent_id} GT')
                
                # Plot predicted positions
                pos_pred = torch.flatten(generated_trajectory, end_dim=1).cpu()
                for agent_id in range(num_agents):
                    pred_xy = pos_pred[:, agent_id]
                    ax1.plot(pred_xy[:, 0], pred_xy[:, 1], color=colors[agent_id],
                            label=f'Agent {agent_id} Pred')
                
                ax1.set_title(f'Pose Trajectories at Epoch {epoch+1}')
                ax1.legend()
                ax1.grid(True)
                
                # Plot ground truth actions
                pos_des = torch.flatten(test_states_actions, end_dim=1).cpu()
                for agent_id in range(num_agents):
                    des_xy = pos_des[:, agent_id]
                    ax2.plot(des_xy[:, 4], des_xy[:, 5], color=colors[agent_id],
                            linestyle='--', label=f'Agent {agent_id} GT')
                
                # Plot predicted actions
                pos_pred = torch.flatten(generated_trajectory, end_dim=1).cpu()
                for agent_id in range(num_agents):
                    pred_xy = pos_pred[:, agent_id]
                    ax2.plot(pred_xy[:, 4], pred_xy[:, 5], color=colors[agent_id],
                            label=f'Agent {agent_id} Pred')
                
                ax2.set_title(f'Action Trajectories at Epoch {epoch+1}')
                ax2.legend()
                ax2.grid(True)
                
                # Add to TensorBoard
                writer.add_figure('Trajectories', fig, epoch)
        
        avg_train_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/Actions/MADP_agent_diffuse/epoch', avg_train_loss, epoch)
        
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.6f}", refresh=False)
        pbar.update()
    
    # Close the TensorBoard writer
    writer.close()
    
    return model

def generate_trajectories_with_curriculum(model, context, device,
                                         start_agents=1, max_agents=4, steps=10):
    """
    Generate trajectories with increasing number of agents to demonstrate curriculum capability
    Args:
        model: Trained diffusion model
        context: Context information for generation
        device: Device to run inference on
        start_agents: Starting number of agents
        max_agents: Maximum number of agents
        steps: Number of steps in the curriculum
    Returns:
        Dictionary mapping number of agents to generated trajectories
    """
    results = {}
    
    # Move context to device
    context = [c.to(device) for c in context]
    batch_size = context[0].shape[0]
    
    # Generate trajectories with increasing number of agents
    for num_agents in range(start_agents, max_agents + 1):
        print(f"Generating trajectories for {num_agents} agents")
        
        # Set active agents
        active_agents = torch.full((batch_size,), num_agents, device=device, dtype=torch.long)
        
        # Generate trajectories
        trajectories = model.sample_trajectory(context, active_agents, save_steps=False)
        
        # Store results
        results[num_agents] = trajectories.cpu()
        
    return results







# ===== Main Function =====

def main():
    # Load the Training configuration and details
    train_config_file = "MADP_training_config.yaml"
    with open(train_config_file, 'r') as file:
        train_config = yaml.safe_load(file)
    
    # Get the scenario name and num of agents
    scenario_name = train_config['param']['scenario']
    num_agents = train_config['param']['num_agents']
    horizon = train_config['param']['horizon']
    cuda_param = train_config['param']['device']
    
    # Create dataset and dataloader
    # Path to HDF5 file
    h5_path = f"{scenario_name}_Na_{num_agents}_T_{horizon}_dataset.h5"
    print(f"Loaded data from {h5_path}")
    
    # Create train and test datasets
    train_dataset = TrajectoryDataset(h5_path=h5_path, split='train', horizon=8)
    test_dataset = TrajectoryDataset(h5_path=h5_path, split='test', horizon=8)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize model
    model = MultiAgentDiffusionPolicy(
        state_dim=6,
        image_channels=3,
        max_num_agents=num_agents,  # Adjust based on your maximum number of agents
        horizon=horizon,  # Set to match your trajectory length
        hidden_dim=32,
        n_diffusion_steps=200,
        noise=None
    )
    
    if train_config['param']['train']:
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Train the model
        trained_model = train_diffusion_model(
            model=model,
            dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            num_epochs=20000,
            device=cuda_param if torch.cuda.is_available() else "cpu"
        )
        
        # Save the trained model
        torch.save(trained_model.state_dict(), "diffusion_policy.pth")
    else:
        model.load_state_dict(torch.load("diffusion_policy.pth"))
    
    test_n_agents = 1  # Generate trajectory for 1 agent
    device = cuda_param if torch.cuda.is_available() else "cpu"
    
    # In the inference section of your code
    if train_config['param']['train']:
        model.load_state_dict(torch.load("diffusion_policy.pth"))
        model.eval()
        
        # Create a test dataloader with your test dataset
        test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)
        
        # Get a single sample
        test_iterator = iter(test_dataloader)
        test_batch = next(test_iterator)
        
        # Extract the components
        test_states_actions = test_batch['states_actions'].to(device)
        test_context = tuple(c.to(device) for c in test_batch['context'])
        
        # Generate trajectory with intermediate steps
        with torch.no_grad():
            generated_trajectory, intermediate_trajectories = model.sample_trajectory(
                test_context,
                test_n_agents,
                save_steps=True,
                step_interval=50
            )
        
        # Create a directory to save plots
        os.makedirs("diffusion_steps", exist_ok=True)
        
        # Plot ground truth for reference
        plt.figure(figsize=(8, 6))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        
        pos_des = torch.flatten(test_states_actions, end_dim=1).to('cpu')
        for agent_id in range(test_n_agents):
            # Desired trajectory
            des_xy = pos_des[:, agent_id]
            plt.plot(des_xy[:, 0], des_xy[:, 1], color=colors[agent_id],
                    linestyle='--', label=f'Agent {agent_id} Ground Truth', linewidth=2)
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Ground Truth Trajectories')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("diffusion_steps/ground_truth.png")
        plt.close()
        
        # Plot intermediate trajectories
        for step, traj in intermediate_trajectories.items():
            plt.figure(figsize=(8, 6))
            
            # Normalize if needed (consistent with your final visualization)
            pos_pred = torch.flatten(traj, end_dim=1)
            
            for agent_id in range(test_n_agents):
                # Predicted trajectory at this step
                pred_xy = pos_pred[:, agent_id]
                plt.scatter(pred_xy[:, 0], pred_xy[:, 1], color=colors[agent_id], alpha=step/len(intermediate_trajectories.items()),
                           label=f'Agent {agent_id}', linewidth=2)
            
            # Add start and goal markers
            start_poses = test_batch['context'][1][0].cpu()
            goal_poses = test_batch['context'][2][0].cpu()
            
            for agent_id in range(test_n_agents):
                plt.scatter(start_poses[agent_id, 0], start_poses[agent_id, 1],
                           color=colors[agent_id], marker='o', s=100, label=f'Start {agent_id}')
                plt.scatter(goal_poses[agent_id, 0], goal_poses[agent_id, 1],
                           color=colors[agent_id], marker='x', s=100, label=f'Goal {agent_id}')
            
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Trajectory at Diffusion Step {step}')
            plt.axis('equal')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"diffusion_steps/step_{step:04d}.png")
            plt.close()
        
        # Create a GIF from the saved images
        try:
            import imageio
            
            # Get all PNG files and sort them by step number
            filenames = sorted(glob.glob("diffusion_steps/step_*.png"))
            
            # Add ground truth at the beginning
            filenames = ["diffusion_steps/ground_truth.png"] + filenames
            
            # Create GIF
            with imageio.get_writer("diffusion_process.gif", mode="I", duration=0.5) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            
            print("Created GIF of diffusion process: diffusion_process.gif")
        except ImportError:
            print("Could not create GIF. Please install imageio: pip install imageio")

if __name__ == "__main__":
    main()
