import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
import pdb
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



from MADP6 import MultiAgentDiffusionPolicy


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
        for  batch_idx, batch in enumerate(dataloader):
            states_actions = batch['states_actions'].to(device)
            context = tuple(c.to(device) for c in batch['context'])
            # pdb.set_trace()
            
            optimizer.zero_grad()
            loss = model(states_actions, context)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            

            # Every few epochs, generate and log a sample trajectory
            # In your training loop visualization code
            if (epoch + 1) % 100 == 0:
            # if epoch == 0:
                with torch.no_grad():
                    test_batch = next(iter(test_dataloader))
                    # Test data
                    test_states_actions = test_batch['states_actions'].to(device)
                    test_context = tuple(c.to(device) for c in test_batch['context'])

                    # Current Data 
                    # test_states_actions = batch['states_actions'].to(device)
                    # test_states_actions = test_states_actions[0]
                    # test_context = tuple(c.to(device) for c in batch['context'])
                    # pdb.set_trace()
                    # test_context = test_context[0].squeeze(0)
                    
                    
                    # Generate trajectory
                    generated_trajectory = model.sample_trajectory(test_context, num_agents, save_steps=False)
                    
                    # Create visualization
                    fig = plt.figure(figsize=(8, 6))
                    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
                    
                    # Plot ground truth (now shorter)
                    pos_des = test_states_actions.reshape(-1, num_agents, test_states_actions.shape[-1]).cpu()
                    for agent_id in range(num_agents):
                        des_xy = pos_des[:, agent_id, :2]  # Just x,y position
                        plt.plot(des_xy[:, 0], des_xy[:, 1], color=colors[agent_id],
                                linestyle='--', label=f'Agent {agent_id} GT')
                    
                    start_poses = test_batch['context'][1][0].cpu()
                    goal_poses = test_batch['context'][2][0].cpu()
                    for agent_id in range(num_agents):
                        plt.scatter(start_poses[agent_id, 0], start_poses[agent_id, 1], 
                                    color=colors[agent_id], marker='o', s=100, label=f'Start {agent_id}')
                        plt.scatter(goal_poses[agent_id, 0], goal_poses[agent_id, 1], 
                                    color=colors[agent_id], marker='x', s=100, label=f'Goal {agent_id}')
                    
                    # Plot predictions
                    pos_pred = generated_trajectory.reshape(-1, num_agents, generated_trajectory.shape[-1]).cpu()
                    for agent_id in range(num_agents):
                        pred_xy = pos_pred[:, agent_id, :2]  # Just x,y position
                        plt.plot(pred_xy[:, 0], pred_xy[:, 1], color=colors[agent_id],
                                label=f'Agent {agent_id} Pred')
                        
                    plt.title(f'Trajectories at Epoch {epoch+1}')
                    plt.legend()
                    plt.grid(True)
                    
                    # Add to TensorBoard
                    writer.add_figure('Trajectories', fig, epoch)


        
        avg_train_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/MADP4/epoch', avg_train_loss, epoch)
        # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.6f}", refresh=False)
        pbar.update()
        
    
    # Close the TensorBoard writer
    writer.close()
    return model



def generate_trajectories(model, context, n_agents, device="cuda"):
    model.to(device)
    model.eval()
    
    # Move context to device
    context = context.to(device)
    
    # Generate trajectories using the model's sampling method
    with torch.no_grad():
        trajectories = model.sample_trajectory(context, n_agents)
    
    return trajectories

    
def test_diffusion_model(model, dataloader, device="cuda"):
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            states_actions = batch['states_actions'].to(device)
            context = tuple(c.to(device) for c in batch['context'])
            
            # Use a fixed timestep for consistent evaluation
            b = states_actions.shape[0]
            t = torch.ones(b, device=device).long() * 500  # Middle of diffusion process
            
            loss = model.p_losses(states_actions, context, t)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.6f}")
    return avg_loss



# Define a custom dataset
class ChunkedTrajectoryDataset(Dataset):
    def __init__(self, h5_path, chunk_size=5, num_samples=None, split='train', train_ratio=0.8):
        self.h5_file = h5py.File(h5_path, 'r')
        self.chunk_size = chunk_size
        
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
        
        # Pre-compute the total number of chunks across all episodes
        self.chunks_per_episode = {}
        self.total_chunks = 0
        
        for idx in self.indices:
            episode_name = f'episode_{idx}'
            trajectory = self.h5_file['trajectories'][episode_name][:]
            num_chunks = (trajectory.shape[0] - 1) // self.chunk_size + 1
            self.chunks_per_episode[idx] = num_chunks
            self.total_chunks += num_chunks
    
    def __len__(self):
        return self.total_chunks
    
    def __getitem__(self, idx):
        # Find which episode and which chunk within that episode
        episode_idx = 0
        chunk_idx = idx

        for i in self.indices:
            if chunk_idx >= self.chunks_per_episode[i]:
                chunk_idx -= self.chunks_per_episode[i]
                episode_idx += 1
            else:
                episode_idx = i
                break

        episode_name = f'episode_{episode_idx}'

        # Load full trajectory data
        full_trajectory = torch.tensor(self.h5_file['trajectories'][episode_name][:], dtype=torch.float32)

        # Calculate start and end indices for this chunk
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, full_trajectory.shape[0])

        # Extract the chunk
        states_actions = full_trajectory[start_idx:end_idx]

        # If the chunk is smaller than chunk_size, repeat the last state/action
        if states_actions.shape[0] < self.chunk_size:
            last_state = states_actions[-1:]
            repeats_needed = self.chunk_size - states_actions.shape[0]
            repeated_last_states = last_state.repeat(repeats_needed, 1, 1)
            states_actions = torch.cat([states_actions, repeated_last_states], dim=0)

        # Define start_pose_context and goal_pose_context
        start_pose_context = states_actions[0, :, :4]  # First timestep, position and velocity
        goal_pose_context = states_actions[-1, :, :2]  # Last timestep, final position

        # Load context data
        frames = torch.tensor(self.h5_file['frames'][episode_name][0:1], dtype=torch.float32)

        num_agents = self.h5_file.attrs['num_agents']

        # Create context tuple
        context = (frames, start_pose_context, goal_pose_context, torch.tensor([num_agents]))

        return {
            'states_actions': states_actions,
            'context': context,
            'chunk_info': {
                'episode_idx': episode_idx,
                'chunk_idx': chunk_idx,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
        }






if __name__=='__main__':
        
    # Load the Training configuration and details
    train_config_file = "MADP_training_config.yaml"
    with open(train_config_file, 'r') as file:
        train_config = yaml.safe_load(file)
    
    # Get the scenario name and num of agents
    scenario_name = train_config['param']['scenario']
    num_agents = train_config['param']['num_agents']
    horizon = train_config['param']['horizon']


    # Create dataset and dataloader
        
    # Path to HDF5 file
    h5_path = f"{scenario_name}_Na_{num_agents}_dataset.h5"

    # Create train and test datasets with chunking
    train_dataset = ChunkedTrajectoryDataset(h5_path=h5_path, chunk_size=horizon, split='train')
    test_dataset = ChunkedTrajectoryDataset(h5_path=h5_path, chunk_size=horizon, split='test')


    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    
    # Initialize model
    model = MultiAgentDiffusionPolicy(
    state_dim=6,                        # Dim of x_i = [s_i, a_i]
    image_channels=3,
    max_num_agents=num_agents,          # Adjust based on your maximum number of agents
    horizon=horizon,                         # Set to match your trajectory length
    hidden_dim=16,
    n_diffusion_steps=100
    )

    # Name and path of model
    timestamp = datetime.now().strftime("%Y%m")
    model_path = f"MADP/diffusion_policy_{timestamp}.pth"
    pdb.set_trace()

    '''
    Training
    '''
    if train_config['param']['train']:
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Train the model
        trained_model = train_diffusion_model(
            model=model,
            dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            num_epochs=100,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        # Save the trained model
        torch.save(trained_model.state_dict(), model_path)

    else: 
        model.load_state_dict(torch.load(model_path))
    

    '''
    Inference
    '''

    test_n_agents = 4  # Generate trajectory for 3 agents
    # test_context = generate_test_case(test_n_agents)  # Single batch with context
    device = "cuda:0"

    # In the inference section of your code
    if train_config['param']['train']:
        model.load_state_dict(torch.load(model_path))
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
                step_interval=50  #This if the interval in Diffusion step after which it stores the noisy trajectory.
            )
        
        # Create a directory to save plots
        import os
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
            # pos_pred = torch.nn.functional.normalize(pos_pred, p=2.0, dim=2)
            
            for agent_id in range(test_n_agents):
                # Predicted trajectory at this step
                pred_xy = pos_pred[:, agent_id]
                plt.scatter(pred_xy[:, 0], pred_xy[:, 1], color=colors[agent_id], alpha=step/250,
                        label=f'Agent {agent_id}', linewidth=2, marker='v')
            
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
            import glob
            
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

