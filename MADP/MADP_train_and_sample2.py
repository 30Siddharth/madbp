import torch
from torch.utils.data import Dataset, DataLoader
from MADP4 import MultiAgentDiffusionPolicy
import numpy as np
import yaml
import pdb
import h5py
from tqdm import tqdm

def train_diffusion_model(model, dataloader, test_dataloader=None, optimizer=None, 
                         num_epochs=100, eval_interval=10, device="cuda"):
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_test_loss = float('inf')
    pbar = tqdm(total=num_epochs, desc="epoch loss = 0")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            states_actions = batch['states_actions'].to(device)
            context = tuple(c.to(device) for c in batch['context'])
            
            optimizer.zero_grad()
            loss = model(states_actions, context)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(dataloader)
        # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.6f}", refresh=False)
        pbar.update()
        
        # Testing phase (if test_dataloader is provided)
        # if test_dataloader is not None and (epoch + 1) % eval_interval == 0:
        #     model.eval()
        #     test_loss = 0.0
        #     with torch.no_grad():
        #         for batch in test_dataloader:
        #             states_actions = batch['states_actions'].to(device)
        #             context = tuple(c.to(device) for c in batch['context'])
        #             loss = model(states_actions, context)
        #             test_loss += loss.item()
            
        #     avg_test_loss = test_loss / len(test_dataloader)
        #     print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {avg_test_loss:.6f}")
            
        #     # Save best model
        #     if avg_test_loss < best_test_loss:
        #         best_test_loss = avg_test_loss
        #         torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'train_loss': avg_train_loss,
        #             'test_loss': avg_test_loss,
        #         }, "best_diffusion_policy.pth")
    
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
class TrajectoryDataset(Dataset):
    def __init__(self, h5_path, num_samples=None, split='train', train_ratio=0.8):
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
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Map idx to the actual episode index
        actual_idx = self.indices[idx]
        episode_name = f'episode_{actual_idx}'
        
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
    scenario_name = train_config['param']['scenario']
    num_agents = train_config['param']['num_agents']


    # Create dataset and dataloader
        
    # Path to HDF5 file
    h5_path = f"{scenario_name}_Na_{num_agents}_dataset.h5"


    # Create train and test datasets
    train_dataset = TrajectoryDataset(h5_path=h5_path, split='train')
    test_dataset = TrajectoryDataset(h5_path=h5_path, split='test')

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    
    # Initialize model
    model = MultiAgentDiffusionPolicy(
        state_dim=6,
        image_channels=3,
        max_num_agents=num_agents,  # Adjust based on your maximum number of agents
        horizon=99,  # Set to match your trajectory length
        hidden_dim=128,
        n_diffusion_steps=500
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
            num_epochs=1000,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Save the trained model
        torch.save(trained_model.state_dict(), "diffusion_policy.pth")

    else: 
        model.load_state_dict(torch.load("diffusion_policy.pth"))
    
    # Inference example
    test_n_agents = 4  # Generate trajectory for 3 agents
    # test_context = generate_test_case(test_n_agents)  # Single batch with context
    device = "cuda"

    # Create a test dataloader with your test dataset
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Get a single sample
    test_iterator = iter(test_dataloader)
    test_batch = next(test_iterator)

    # Extract the components
    test_states_actions = test_batch['states_actions'].to(device)
    test_context = tuple(c.to(device) for c in test_batch['context'])
    

    # Generate trajectory
    with torch.no_grad():
        generated_trajectory = model.sample_trajectory(test_context, test_n_agents)

    # Visualize or compare with ground truth
    
    print(f"Generated trajectory shape: {generated_trajectory.shape}")
    print(f"Ground truth shape: {test_states_actions.shape}")
    
    pos_pred = torch.flatten(generated_trajectory, end_dim=1).to('cpu')
    pos_des = torch.flatten(test_states_actions, end_dim=1).to('cpu')
    pos_pred = torch.nn.functional.normalize(pos_pred, p=2.0, dim = 2)
    pdb.set_trace()
    # C = A-B
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for agent_id in range(4):
        # Predicted trajectory
        pred_xy = pos_pred[:, agent_id]
        plt.plot(pred_xy[:, 0], pred_xy[:, 1], color=colors[agent_id], label=f'Agent {agent_id} Pred', linewidth=2)

        # Desired trajectory
        des_xy = pos_des[:, agent_id]
        plt.plot(des_xy[:, 0], des_xy[:, 1], color=colors[agent_id], linestyle='--', label=f'Agent {agent_id} Des', linewidth=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Predicted vs Desired Trajectories (All Agents)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # # Generate trajectories
    # generated_trajectories = generate_trajectories(
    #     model=trained_model,
    #     context=test_context,
    #     n_agents=test_n_agents,
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )

    # print(f"Generated trajectories shape: {generated_trajectories.shape}")
    # Should output: (1, 10, 3, 6) - (batch_size, horizon, n_agents, state_dim)
