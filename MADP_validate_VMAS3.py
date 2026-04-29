import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import vmas
from vmas import make_env
from vmas.simulator.utils import save_video
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from PIL import Image  # Added for VMAS-style frame storage

# Import from your v2_1 modules (matching the reference file)
from MADP_diffusion_v2_1 import EnhancedMultiAgentDiffusionModel
from MADP_train_and_sample_v2_1 import NormalizedTrajectoryDataset, collate_fn

import pdb

class TrajectoryExecutor:
    def __init__(self, scenario_name=None, num_envs=1, max_steps=40, n_agents=4, device='cuda', 
                 test_start_positions=None, test_goal_positions=None):
        """
        Initialize VMAS Navigation environment for trajectory execution
        """
        self.device = device
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.n_agents = n_agents # Actual active agents in VMAS
        self.max_agents = 10 # Fixed architecture requirement

        self.test_start_positions = test_start_positions
        self.test_goal_positions = test_goal_positions

        # Create VMAS Navigation environment
        self.env = make_env(
            scenario=scenario_name, # Use standard navigation scenario
            num_envs=num_envs,
            device=device,
            continuous_actions=True,
            max_steps=max_steps,
            # Navigation scenario specific kwargs
            n_agents=n_agents, # Use actual number of agents for VMAS
            collision_reward=-1.0,
            dist_shaping_factor=1.0,
            final_reward=0.01,
            agent_radius=0.05,
            x_semidim=1.0,
            y_semidim=1.0,
            lidar_range=0.1,
            shared_reward=True,
            use_test_positions=True,
            test_start_positions=self.test_start_positions,
            test_goal_positions=self.test_goal_positions

        )

        # Controller parameters
        self.Kp = 50.0 # Position gain
        self.Kv = 4.5 # Velocity damping
        self.max_force = 1.0 # Maximum force limit in VMAS

    def smooth_trajectory(self, trajectory, method='cubic_spline', strength=0.2):
        """Smooth trajectory using cubic spline interpolation"""
        T, Na, D = trajectory.shape
        smoothed = trajectory.clone()
        
        for agent_idx in range(Na):
            agent_traj = trajectory[:, agent_idx].cpu().numpy() # [T, 2]
            if method == 'cubic_spline':
                t_points = np.linspace(0, 1, T)
                for dim in range(D):
                    # Fit cubic spline
                    cs = CubicSpline(t_points, agent_traj[:, dim], bc_type='natural')
                    smoothed_values = cs(t_points)
                    # Blend with original based on strength
                    final_values = (1 - strength) * agent_traj[:, dim] + strength * smoothed_values
                    # Preserve start and end points
                    final_values[0] = agent_traj[0, dim]
                    final_values[-1] = agent_traj[-1, dim]
                    smoothed[:, agent_idx, dim] = torch.tensor(final_values, device=trajectory.device)
        
        return smoothed

    def compute_control_forces(self, current_states, target_positions, target_velocities=None):
        """Compute control forces using proportional-derivative control"""
        # Extract current positions and velocities from VMAS (only active agents)
        current_pos = torch.stack([agent.state.pos for agent in self.env.agents], dim=1) # [num_envs, n_agents, 2]
        current_vel = torch.stack([agent.state.vel for agent in self.env.agents], dim=1) # [num_envs, n_agents, 2]

        # Position error (only for active agents)
        pos_error = target_positions - current_pos

        # Velocity error (if target velocities provided)
        if target_velocities is not None:
            vel_error = target_velocities - current_vel
        else:
            # Estimate target velocity from position error
            vel_error = -current_vel # Simple damping

        # PD control law
        forces = self.Kp * pos_error + self.Kv * vel_error

        # Clamp forces to VMAS limits
        forces = torch.clamp(forces, -self.max_force, self.max_force)

        return forces

    def rendering_callback(self, env, step_data=None):
        """VMAS-style rendering callback that stores frames as PIL Images"""
        frame = env.render(mode="rgb_array", agent_index_focus=None, visualize_when_rgb=True)
        if frame is not None:
            # Convert to PIL Image following VMAS tutorial pattern
            env.frames.append(Image.fromarray(frame))

    def execute_complete_trajectory(self, trajectory, render=True, store_frames=True):
        """Execute COMPLETE trajectory in VMAS environment using VMAS callback pattern"""
        T, Na, D = trajectory.shape
        assert Na <= self.n_agents, f"Trajectory has {Na} agents but environment supports {self.n_agents}"
        
        print(f"Executing COMPLETE trajectory with {T} timesteps for {Na} agents")

        # Smooth the COMPLETE trajectory
        smoothed_traj = self.smooth_trajectory(trajectory, method='velocity_smoothing', strength=0.2)

        # Reset environment
        obs = self.env.reset()

        # Storage for results
        executed_positions = []
        tracking_errors = []
        rewards_history = []
        
        # Initialize frames storage following VMAS tutorial pattern
        if store_frames:
            self.env.frames = []

        # Execute COMPLETE trajectory step by step
        for t in range(min(T-1, self.max_steps-1)):
            # Get target positions for current timestep (only active agents)
            target_pos = smoothed_traj[t+1, :Na].unsqueeze(0).to(self.device) # [1, Na, 2]

            # Compute target velocities (finite difference)
            if t < T-2:
                target_vel = (smoothed_traj[t+2, :Na] - smoothed_traj[t, :Na]).unsqueeze(0).to(self.device) / 2.0
            else:
                target_vel = torch.zeros_like(target_pos)

            # Compute control forces
            forces = self.compute_control_forces(None, target_pos, target_vel)

            # Create actions as list of per-agent tensors with proper shape
            batch_actions = torch.zeros(self.n_agents, 2, device=self.device)
            for agent_idx in range(self.n_agents):
                if agent_idx < Na:
                    batch_actions[agent_idx] = forces[0, agent_idx]
                # else: already zeros from initialization

            # Convert to list of per-agent tensors with proper shape
            actions = [batch_actions[i].unsqueeze(0) for i in range(self.n_agents)]

            # Step environment
            obs, rewards, dones, info = self.env.step(actions)

            # Call rendering callback following VMAS pattern
            if store_frames and render:
                self.rendering_callback(self.env)

            # Record current positions (only active agents)
            current_positions = torch.stack([agent.state.pos for agent in self.env.agents], dim=1) # [1, n_agents, 2]
            executed_positions.append(current_positions[0, :Na].cpu()) # [Na, 2]

            # Compute tracking error
            tracking_error = torch.norm(current_positions[0, :Na] - target_pos[0], dim=-1).mean()
            tracking_errors.append(tracking_error.cpu().item())

            # Record rewards
            total_reward = sum([r.sum().item() for r in rewards])
            rewards_history.append(total_reward)

            # Check if done
            if dones.any():
                print(f"Episode finished at step {t+1}")
                break

        # Convert results to tensors
        executed_positions = torch.stack(executed_positions) # [T_executed, Na, 2]

        print(f"Stored {len(self.env.frames) if hasattr(self.env, 'frames') else 0} PIL Image frames")

        return {
            'executed_positions': executed_positions,
            'planned_positions': smoothed_traj[:len(executed_positions), :Na],
            'tracking_errors': tracking_errors,
            'rewards': rewards_history,
            'frames': self.env.frames if hasattr(self.env, 'frames') else None
        }

    def create_gif_from_frames(self, results, gif_name="trajectory_execution.gif"):
        """Create GIF from stored PIL Image frames using VMAS tutorial method"""
        if results['frames'] and len(results['frames']) > 0:
            # Use exact VMAS tutorial method for GIF creation
            results['frames'][0].save(
                gif_name,
                save_all=True,
                append_images=results['frames'][1:],
                duration=1,  # milliseconds per frame (adjust for speed)
                loop=1,  # infinite loop
            )
            print(f"GIF saved as {gif_name}")
            return True
        else:
            print("No frames to save")
            return False

    def evaluate_performance(self, results):
        """Evaluate trajectory execution performance"""
        executed = results['executed_positions']
        planned = results['planned_positions']
        planned = planned.detach().cpu().numpy()

        # Compute metrics
        position_errors = torch.norm(executed - planned, dim=-1) # [T, Na]
        mean_tracking_error = position_errors.mean().item()
        max_tracking_error = position_errors.max().item()
        final_position_error = position_errors[-1].mean().item()
        total_reward = sum(results['rewards'])

        metrics = {
            'mean_tracking_error': mean_tracking_error,
            'max_tracking_error': max_tracking_error,
            'final_position_error': final_position_error,
            'total_reward': total_reward,
            'success_rate': 1.0 if final_position_error < 0.1 else 0.0
        }

        return metrics

    def plot_trajectories(self, results, plot_name="trajectory_comparison.png"):
        """Plot executed vs planned trajectories with error"""
        executed = results['executed_positions']
        planned = results['planned_positions']
        errors = results['tracking_errors']

        planned = planned.detach().cpu().numpy() 
        
        plt.figure(figsize=(15, 10))
        
        # Trajectory Comparison
        plt.subplot(1, 2, 1)
        for i in range(executed.shape[1]):
            plt.plot(executed[:, i, 0], executed[:, i, 1], 'o-', label=f'Agent {i} Executed')
            plt.plot(planned[:, i, 0], planned[:, i, 1], 'x--', label=f'Agent {i} Planned')
        plt.title('Trajectory Comparison')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')    
        
        # Error Over Time
        plt.subplot(1, 2, 2)
        plt.plot(errors, 'r-', linewidth=2)
        plt.title('Tracking Error Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Error (m)')
        plt.grid(True)
        
        # plt.tight_layout()
        plt.savefig(plot_name)
        plt.close()
        return plot_name

def execute_complete_diffusion_trajectory_in_vmas(scenario_name, model_path=None, device='cuda'):
    """Execute COMPLETE diffusion-generated trajectories in VMAS with proper 10-agent architecture"""
    
    # Load configuration
    train_config_file = "MADP_training_config.yaml"
    try:
        with open(train_config_file, 'r') as file:
            train_config = yaml.safe_load(file)
        scenario_name = train_config['param']['scenario']
        num_agents = train_config['param']['num_agents']
        full_horizon = train_config['param']['horizon'] # COMPLETE horizon
        h5_path = f"{scenario_name}_Na_{num_agents}_T_{full_horizon}_dataset.h5"
        diffusion_steps = train_config['param']['diffuse_steps']
    except FileNotFoundError:
        scenario_name = "default"
        num_agents = 4
        full_horizon = 40 # COMPLETE horizon
        h5_path = "dataset.h5"
        diffusion_steps = 500

    print(f"Executing COMPLETE MADP trajectories in VMAS Navigation environment")
    print(f"Scenario: {scenario_name}, Active Agents: {num_agents}, COMPLETE Horizon: {full_horizon}")
    print(f"Model Architecture: Fixed to 10 agents for context computation")
    print(f"Dataset: {h5_path}")


    # CRITICAL FIX: Use proper data loading with collate_fn
    dataset = NormalizedTrajectoryDataset(h5_path, split='test', horizon=full_horizon)
    test_loader = DataLoader(dataset, 1, shuffle=True, collate_fn=collate_fn, pin_memory=True)

    # Load model with correct parameters matching your v2_1 reference
    model = EnhancedMultiAgentDiffusionModel(
        max_agents=10, # Fixed architecture requirement
        horizon=full_horizon, # Use COMPLETE horizon for model (matches v2_1)
        state_dim=2,
        img_ch=3,
        hid=128,
        diffusion_steps=diffusion_steps, # Match your v2_1 training config
        schedule_type='linear' # Match your v2_1 training config
    ).to(device)

    # Load trained weights
    if model_path is None:
        model_path = f"enhanced_madp_{scenario_name}.pth" # Match your original naming

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first.")
        return None, None

    # Generate COMPLETE trajectory
    model.eval()
    with torch.no_grad():
        frame, start, goal, na, full_traj = next(iter(test_loader))
        frame = frame.to(device)
        start = start.to(device) # Now [1, 10, 2]
        goal = goal.to(device) # Now [1, 10, 2]
        na = na.to(device)

        print(f"Generating COMPLETE trajectory for {na.item()} active agents (out of 10 total)")
        print(f"COMPLETE horizon length: {full_horizon}")

        # Extract test positions for VMAS scenario (only active agents)
        n_active = na.item()
        test_start_positions = start[0, :n_active].cpu().numpy()  # [n_active, 2]
        test_goal_positions = goal[0, :n_active].cpu().numpy()   # [n_active, 2]
        
        print(f"Test start positions: {test_start_positions}")
        print(f"Test goal positions: {test_goal_positions}")
        

        print(f"Generating COMPLETE trajectory for {na.item()} active agents (out of 10 total)")
        print(f"COMPLETE horizon length: {full_horizon}")

        # Verify 10-agent architecture
        assert start.shape[1] == 10, f"Start positions should be padded to 10 agents, got {start.shape[1]}"
        assert goal.shape[1] == 10, f"Goal positions should be padded to 10 agents, got {goal.shape[1]}"

        # Generate COMPLETE trajectory using the method from your v2_1 file
        predicted_trajectory = model.sample_with_constraints_and_smooth(
            frame, start, goal, na,
            steps=diffusion_steps, # Use fewer steps for faster sampling
            smoothing_method='cubic_spline',
            max_step_size=0.2
        )

        # Denormalize and extract active agents only
        xy_mean = dataset.xy_mean.to(device)
        xy_std = dataset.xy_std.to(device)
        trajectory_denorm = predicted_trajectory * (3 * xy_std) + xy_mean
        n_active = na.item()
        trajectory = trajectory_denorm[0, :n_active].permute(1, 0, 2) # [T_complete, Na_active, 2]

        print(f"COMPLETE trajectory shape: {trajectory.shape}") # Should be [full_horizon, n_active, 2]

    # Execute COMPLETE trajectory in VMAS
    # Initialize executor for COMPLETE horizon
    vmas_scenario_name = f"{scenario_name}_test"   # Scanrio name for VMAS TEST CASE

    executor = TrajectoryExecutor(
        scenario_name=vmas_scenario_name, 
        num_envs=1,
        max_steps=full_horizon , # Allow for COMPLETE trajectory
        n_agents=num_agents, # Actual active agents
        device=device,
        test_start_positions=test_start_positions,  # Pass test positions
        test_goal_positions=test_goal_positions     # Pass test positions
    )

    print("Executing COMPLETE trajectory in VMAS Navigation environment...")
    results = executor.execute_complete_trajectory(
        trajectory, # COMPLETE trajectory
        render=True,
        store_frames=True  # Store frames using VMAS pattern
    )

    # Create GIF using VMAS method
    gif_name = f"madp_{scenario_name}_COMPLETE_validation.gif"
    executor.create_gif_from_frames(results, gif_name)

    plot_name = f"madp_{scenario_name}_trajectory_plot.png"
    executor.plot_trajectories(results, plot_name)

    # Evaluate performance
    metrics = executor.evaluate_performance(results)

    print("\nCOMPLETE Trajectory Execution Results:")
    print(f"Active Agents: {n_active}")
    print(f"Mean Tracking Error: {metrics['mean_tracking_error']:.4f}")
    print(f"Max Tracking Error: {metrics['max_tracking_error']:.4f}")
    print(f"Final Position Error: {metrics['final_position_error']:.4f}")
    print(f"Total Reward: {metrics['total_reward']:.4f}")
    print(f"Success Rate: {metrics['success_rate']:.2f}")

    return results, metrics

# Usage example
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Execute COMPLETE trajectory
    results, metrics = execute_complete_diffusion_trajectory_in_vmas(
        "navigation_v3_test",
        model_path="enhanced_madp_navigation_v3.pth",
        device=device
    )

    if results is not None:
        print(f"\nCOMPLETE trajectory execution completed successfully!")
        print(f"GIF created using VMAS tutorial method")
        
        # You can also create additional GIFs with different settings
        # executor.create_gif_from_frames(results, "slow_motion.gif", duration=200)
    else:
        print("Execution failed. Please check the configuration and model files.")
