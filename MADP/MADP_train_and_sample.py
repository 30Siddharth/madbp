import torch
from torch.utils.data import Dataset, DataLoader
from MADP4 import MultiAgentDiffusionPolicy
import numpy as np
import yaml
import pdb
import h5py
from tqdm import tqdm

def train_diffusion_model(model, dataloader, optimizer, num_epochs=100, device="cuda"):
    model.to(device)
    model.train()

    pbar = tqdm(total=num_epochs, desc="epoch loss = 0")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # Get data from batch
            states_actions = batch['states_actions'].to(device)  # Shape: (batch_size, horizon, n_agents, state_dim)
            # context = batch['context'].to(device)  # Shape: (batch_size, context_dim)
            context = tuple(c.to(device) for c in batch['context'])
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass (model's forward method handles random timestep selection)
            loss = model(states_actions, context)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.6f}")
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.6f}", refresh=False)
        pbar.update()
    
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

    
def generate_test_case(test_n_agents):
    

    return test_context


# Define a custom dataset
class TrajectoryDataset(Dataset):
    def __init__(self, h5_path, num_samples=1000):
        self.h5_file = h5py.File(h5_path, 'r')
        self.num_samples = len(self.h5_file['frames'])
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Get episode name
        episode_name = f'episode_{idx}'
        
        # Load data from HDF5 file
        frames = torch.tensor(self.h5_file['frames'][episode_name][:], dtype=torch.float32)
        states_actions = torch.tensor(self.h5_file['trajectories'][episode_name][:], dtype=torch.float32)
        start_poses = torch.tensor(self.h5_file['start_poses'][episode_name][:], dtype=torch.float32)
        goal_poses = torch.tensor(self.h5_file['goal_poses'][episode_name][:], dtype=torch.float32)
        num_agents = self.h5_file.attrs['num_agents']
        
        # Create context tuple properly
        context = (frames, start_poses, goal_poses, torch.tensor([num_agents]))
        
        return {
            'states_actions': states_actions,
            'context': context
        }


if __name__=='__main__':
        
    # Load the Training configuration and details
    train_config_file = "MADP_training_config.yaml"
    with open(train_config_file, 'r') as file:
        train_config = yaml.safe_load(file)
    
    # Get the scenario name and num of agents
    scenario_name = train_config['train']['scenario']
    num_agents = train_config['train']['num_agents']


    # Create dataset and dataloader
        
    # Path to HDF5 file
    h5_path = f"{scenario_name}_Na_{num_agents}_dataset.h5"

    # Create dataset and dataloader
    dataset = TrajectoryDataset(h5_path=h5_path)
    pdb.set_trace()
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = MultiAgentDiffusionPolicy(
        state_dim=6,
        image_channels=3,
        max_num_agents=num_agents,  # Adjust based on your maximum number of agents
        horizon=100,  # Set to match your trajectory length
        hidden_dim=256,
        n_diffusion_steps=1000
    )


    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    trained_model = train_diffusion_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Save the trained model
    torch.save(trained_model.state_dict(), "diffusion_policy.pth")

    # Inference example
    test_n_agents = 4  # Generate trajectory for 3 agents
    test_context = generate_test_case(test_n_agents)  # Single batch with context
    

    # Generate trajectories
    generated_trajectories = generate_trajectories(
        model=trained_model,
        context=test_context,
        n_agents=test_n_agents,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Generated trajectories shape: {generated_trajectories.shape}")
    # Should output: (1, 10, 3, 6) - (batch_size, horizon, n_agents, state_dim)
