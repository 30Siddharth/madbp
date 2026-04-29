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
from PIL import Image
import pdb

# Import from your v2_2 modules (moving horizon version)
from MADP_diffusion_v2_2 import EnhancedMultiAgentDiffusionModel
from MADP_train_and_sample_v2_2 import NormalizedTrajectoryDataset, collate_fn

class MovingHorizonTrajectoryExecutor:
    def __init__(self, scenario_name=None, num_envs=1, max_steps=40, n_agents=4, device='cuda',
                 test_start_positions=None, test_goal_positions=None, horizon_size=8, use_test_positions=False):
        """
        Initialize VMAS Navigation environment for moving horizon trajectory execution
        """
        self.device = device
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.n_agents = n_agents  # Actual active agents in VMAS
        self.max_agents = 10  # Fixed architecture requirement
        self.horizon_size = horizon_size  # Moving horizon window size
        self.test_start_positions = test_start_positions
        self.test_goal_positions = test_goal_positions
        self.use_test_positions = use_test_positions
        
        # Create VMAS Navigation environment
        self.env = make_env(
            scenario=scenario_name,
            num_envs=num_envs,
            device=device,
            continuous_actions=True,
            max_steps=max_steps,
            # Navigation scenario specific kwargs
            n_agents=n_agents,
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
        self.Kp = 30.0  # Position gain
        self.Kv = 2.5   # Velocity damping
        self.max_force = 1.0  # Maximum force limit in VMAS

    def smooth_trajectory_segment(self, trajectory_segment, method='gaussian', strength=0.2):
        """Smooth a single trajectory segment (one horizon window)"""
        T, Na, D = trajectory_segment.shape
        smoothed = trajectory_segment.clone()
        
        for agent_idx in range(Na):
            agent_traj = trajectory_segment[:, agent_idx].cpu().numpy()  # [T, 2]
            
            if method == 'cubic_spline' and T > 3:  # Need at least 4 points for cubic spline
                t_points = np.linspace(0, 1, T)
                for dim in range(D):
                    cs = CubicSpline(t_points, agent_traj[:, dim], bc_type='natural')
                    smoothed_values = cs(t_points)
                    # Blend with original based on strength
                    final_values = (1 - strength) * agent_traj[:, dim] + strength * smoothed_values
                    # Preserve start and end points
                    final_values[0] = agent_traj[0, dim]
                    final_values[-1] = agent_traj[-1, dim]
                    smoothed[:, agent_idx, dim] = torch.tensor(final_values, device=trajectory_segment.device)
            elif method == 'velocity_smoothing':
                # Simple velocity-based smoothing for short segments
                for dim in range(D):
                    velocities = np.diff(agent_traj[:, dim])
                    # Apply moving average to velocities
                    if len(velocities) > 1:
                        smoothed_velocities = np.convolve(velocities, np.ones(min(3, len(velocities)))/min(3, len(velocities)), mode='same')
                        # Reconstruct positions
                        smoothed_positions = np.cumsum(np.concatenate([[agent_traj[0, dim]], smoothed_velocities]))
                        smoothed[:, agent_idx, dim] = torch.tensor(smoothed_positions, device=trajectory_segment.device)
        
        return smoothed

    def compute_control_forces(self, current_states, target_positions, target_velocities=None):
        """Compute control forces using proportional-derivative control"""
        # Extract current positions and velocities from VMAS (only active agents)
        current_pos = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)  # [num_envs, n_agents, 2]
        current_vel = torch.stack([agent.state.vel for agent in self.env.agents], dim=1)  # [num_envs, n_agents, 2]
        
        # Position error (only for active agents)
        pos_error = target_positions - current_pos
        
        # Velocity error (if target velocities provided)
        if target_velocities is not None:
            vel_error = target_velocities - current_vel
        else:
            # Estimate target velocity from position error
            vel_error = -current_vel  # Simple damping
        
        # PD control law
        forces = self.Kp * pos_error + self.Kv * vel_error
        
        # Clamp forces to VMAS limits
        forces = torch.clamp(forces, -self.max_force, self.max_force)
        
        return forces

    def rendering_callback(self, env, step_data=None):
        """VMAS-style rendering callback that stores frames as PIL Images"""
        frame = env.render(mode="rgb_array", agent_index_focus=None, visualize_when_rgb=True)
        if frame is not None:
            env.frames.append(Image.fromarray(frame))

    def execute_moving_horizon_trajectory(self, model, frames, starts, goals, n_agents, 
                                        full_horizon_length, render=True, store_frames=True):
        """Execute trajectory using moving horizon approach with real-time generation"""
        print(f"Executing MOVING HORIZON trajectory with {full_horizon_length} total timesteps")
        print(f"Horizon window size: {self.horizon_size}")
        print(f"Active agents: {n_agents.item()}")
        
        # Reset environment
        obs = self.env.reset()
        
        # Storage for results
        executed_positions = []
        planned_segments = []
        tracking_errors = []
        rewards_history = []
        generation_times = []
        
        # Initialize frames storage
        if store_frames:
            self.env.frames = []
        
        # Current positions start from environment reset
        current_positions = starts.clone()  # [1, 10, 2]
        
        # Calculate number of horizons needed
        # num_horizons = full_horizon_length // self.horizon_size
        # remaining_steps = full_horizon_length
        num_horizons = self.max_steps //self.horizon_size
        remaining_steps = self.max_steps
        
        for horizon_idx in range(num_horizons):
            print(f"Processing horizon {horizon_idx + 1}/{num_horizons}")
            
            # Determine current horizon length
            current_horizon_length = min(self.horizon_size, remaining_steps)
            horizon_goals = goals 
            
            # Set intermediate goals for non-final horizons
            # if horizon_idx == num_horizons - 1:
            #     horizon_goals = goals
            # else:
            #     # Interpolate toward final goals
            #     progress = (horizon_idx + 1) / num_horizons
            #     horizon_goals = current_positions + progress * (goals - current_positions)
            
            # Generate trajectory segment using model
            import time
            start_time = time.time()
            
            with torch.no_grad():
                # Use the model's sampling method for current horizon
                horizon_prediction = model.sample_with_constraints(
                    frames, current_positions, horizon_goals, n_agents,
                    steps=150,  # Reduced steps for real-time performance
                    max_step_size=0.05
                )  # Returns [1, 10, horizon_size, 2]
            # pdb.set_trace()
            generation_time = time.time() - start_time
            generation_times.append(generation_time/self.horizon_size)
            
            # Extract active agents and convert to execution format
            n_active = n_agents.item()
            horizon_segment = horizon_prediction[0, :n_active].permute(1, 0, 2)  # [horizon_size, n_active, 2]
            
            # Smooth the segment
            smoothed_segment = self.smooth_trajectory_segment(
                horizon_segment, method='cubic', strength=0.8
            )
            
            planned_segments.append(smoothed_segment.cpu())
            
            # Execute this horizon segment
            for t in range(min(current_horizon_length - 1, self.max_steps - len(executed_positions) - 1)):
                # Get target positions for current timestep
                target_pos = smoothed_segment[t + 1, :n_active].unsqueeze(0).to(self.device)  # [1, n_active, 2]
                
                # Compute target velocities
                if t < current_horizon_length - 2:
                    target_vel = (smoothed_segment[t + 2, :n_active] - smoothed_segment[t, :n_active]).unsqueeze(0).to(self.device) / 2.0
                else:
                    target_vel = torch.zeros_like(target_pos)
                
                # Compute control forces
                forces = self.compute_control_forces(None, target_pos, target_vel)
                
                # Create actions
                batch_actions = torch.zeros(self.n_agents, 2, device=self.device)
                for agent_idx in range(self.n_agents):
                    if agent_idx < n_active:
                        batch_actions[agent_idx] = forces[0, agent_idx]
                
                actions = [batch_actions[i].unsqueeze(0) for i in range(self.n_agents)]
                
                # Step environment
                obs, rewards, dones, info = self.env.step(actions)
                
                # Rendering
                if store_frames and render:
                    self.rendering_callback(self.env)
                
                # Record current positions
                current_env_positions = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)  # [1, n_agents, 2]
                executed_positions.append(current_env_positions[0, :n_active].cpu())  # [n_active, 2]
                
                # Compute tracking error
                tracking_error = torch.norm(current_env_positions[0, :n_active] - target_pos[0], dim=-1).mean()
                tracking_errors.append(tracking_error.cpu().item())
                
                # Record rewards
                total_reward = sum([r.sum().item() for r in rewards])
                rewards_history.append(total_reward)
                
                # Check if done
                if dones.any():
                    print(f"Episode finished at step {len(executed_positions)}")
                    break
            
            # Update current positions for next horizon (from actual environment state)
            current_env_positions = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)  # [1, n_agents, 2]
            
            # Update current_positions for model input (pad to 10 agents)
            current_positions = torch.zeros_like(starts)  # [1, 10, 2]
            current_positions[0, :n_active] = current_env_positions[0, :n_active]
            
            remaining_steps -= current_horizon_length
            
            if remaining_steps <= 0 or dones.any():
                break
        
        # Convert results to tensors
        executed_positions = torch.stack(executed_positions) if executed_positions else torch.empty(0, n_active, 2)
        
        print(f"Generated {len(planned_segments)} horizon segments")
        print(f"Average generation time per horizon: {np.mean(generation_times):.3f}s")
        print(f"Stored {len(self.env.frames) if hasattr(self.env, 'frames') else 0} frames")
        
        return {
            'executed_positions': executed_positions,
            'planned_segments': planned_segments,
            'tracking_errors': tracking_errors,
            'rewards': rewards_history,
            'frames': self.env.frames if hasattr(self.env, 'frames') else None,
            'generation_times': generation_times,
            'num_horizons': len(planned_segments)
        }

    def create_gif_from_frames(self, results, gif_name="moving_horizon_execution.gif"):
        """Create GIF from stored PIL Image frames"""
        if results['frames'] and len(results['frames']) > 0:
            results['frames'][0].save(
                gif_name,
                save_all=True,
                append_images=results['frames'][1:],
                duration=100,  # Faster for moving horizon
                loop=1
            )
            print(f"GIF saved as {gif_name}")
            return True
        else:
            print("No frames to save")
            return False

    def evaluate_moving_horizon_performance(self, results):
        """Evaluate moving horizon trajectory execution performance"""
        executed = results['executed_positions']
        
        if len(executed) == 0:
            return {'error': 'No execution data available'}
        
        # Reconstruct planned trajectory from segments
        planned_full = torch.cat([seg[:-1] for seg in results['planned_segments'][:-1]] + 
                                [results['planned_segments'][-1]], dim=0)
        
        # Trim to match executed length
        min_length = min(len(executed), len(planned_full))
        executed = executed[:min_length]
        planned_full = planned_full[:min_length]
        
        # Compute metrics
        position_errors = torch.norm(executed - planned_full, dim=-1)  # [T, Na]
        mean_tracking_error = position_errors.mean().item()
        max_tracking_error = position_errors.max().item()
        final_position_error = position_errors[-1].mean().item() if len(position_errors) > 0 else float('inf')
        total_reward = sum(results['rewards'])
        
        # Moving horizon specific metrics
        avg_generation_time = np.mean(results['generation_times'])
        total_horizons = results['num_horizons']
        
        metrics = {
            'mean_tracking_error': mean_tracking_error,
            'max_tracking_error': max_tracking_error,
            'final_position_error': final_position_error,
            'total_reward': total_reward,
            'success_rate': 1.0 if final_position_error < 0.05 else 0.0,  # Slightly relaxed for moving horizon
            'avg_generation_time': avg_generation_time,
            'total_horizons': total_horizons,
            'real_time_capable': avg_generation_time < 0.4  # 100ms threshold for real-time
        }
        
        return metrics

    def plot_moving_horizon_trajectories(self, results, plot_name="moving_horizon_comparison.png"):
        """Plot executed vs planned trajectories with horizon boundaries"""
        executed = results['executed_positions']
        planned_segments = results['planned_segments']
        errors = results['tracking_errors']
        
        if len(executed) == 0:
            print("No execution data to plot")
            return None
        
        # Reconstruct full planned trajectory
        planned_full = torch.cat([seg[:-1] for seg in planned_segments[:-1]] + [planned_segments[-1]], dim=0)
        min_length = min(len(executed), len(planned_full))
        executed = executed[:min_length]
        planned_full = planned_full[:min_length]
        
        plt.figure(figsize=(18, 12))
        
        # Trajectory Comparison with Horizon Boundaries
        plt.subplot(2, 2, 1)
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i in range(executed.shape[1]):
            plt.plot(executed[:, i, 0], executed[:, i, 1], 'o-', 
                    color=colors[i % len(colors)], label=f'Agent {i} Executed', markersize=3)
            plt.plot(planned_full[:, i, 0], planned_full[:, i, 1], 'x--', 
                    color=colors[i % len(colors)], label=f'Agent {i} Planned', alpha=0.7)
        
        # Mark horizon boundaries
        horizon_boundaries = [self.horizon_size * (i + 1) - 1 for i in range(len(planned_segments) - 1)]
        for boundary in horizon_boundaries:
            if boundary < len(executed):
                plt.axvline(x=executed[boundary, 0, 0].item(), color='gray', linestyle=':', alpha=0.5)
        
        plt.title('Moving Horizon Trajectory Comparison')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Error Over Time
        plt.subplot(2, 2, 2)
        plt.plot(errors, 'r-', linewidth=2)
        plt.title('Tracking Error Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Error (m)')
        plt.grid(True)
        
        # Generation Time per Horizon
        plt.subplot(2, 2, 3)
        plt.bar(range(len(results['generation_times'])), results['generation_times'])
        plt.axhline(y=0.1, color='r', linestyle='--', label='Real-time threshold (100ms)')
        plt.title('Generation Time per Horizon')
        plt.xlabel('Horizon Index')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.grid(True)
        
        # Cumulative Reward
        plt.subplot(2, 2, 4)
        cumulative_rewards = np.cumsum(results['rewards'])
        plt.plot(cumulative_rewards, 'g-', linewidth=2)
        plt.title('Cumulative Reward')
        plt.xlabel('Time Step')
        plt.ylabel('Cumulative Reward')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plot_name, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_name


def execute_moving_horizon_diffusion_trajectory_in_vmas(scenario_name, model_path=None, device='cuda'):
    """Execute moving horizon diffusion-generated trajectories in VMAS"""
    
    # Load configuration
    train_config_file = "MADP_training_config.yaml"
    try:
        with open(train_config_file, 'r') as file:
            train_config = yaml.safe_load(file)
        scenario_name = train_config['param']['scenario']
        num_agents = train_config['param']['num_agents']
        full_horizon = train_config['param']['horizon']
        h5_path = f"{scenario_name}_Na_{num_agents}_T_{full_horizon}_dataset.h5"
        diffusion_steps = train_config['param']['diffuse_steps']
        horizon_size = 8  # Fixed horizon size for v2_2
    except FileNotFoundError:
        scenario_name = "default"
        num_agents = 4
        full_horizon = 40
        h5_path = "dataset.h5"
        diffusion_steps = 500
        horizon_size = 8
    
    print(f"Executing MOVING HORIZON MADP trajectories in VMAS Navigation environment")
    print(f"Scenario: {scenario_name}, Active Agents: {num_agents}")
    print(f"Full Horizon: {full_horizon}, Moving Horizon Size: {horizon_size}")
    print(f"Dataset: {h5_path}")
    
    # Load dataset
    dataset = NormalizedTrajectoryDataset(h5_path, split='test', horizon=full_horizon)
    test_loader = DataLoader(dataset, 1, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    
    # Load model with horizon_size (not full_horizon) for v2_2
    model = EnhancedMultiAgentDiffusionModel(
        max_agents=10,
        horizon=horizon_size,  # CRITICAL: Use horizon_size for v2_2 moving horizon
        state_dim=2,
        img_ch=3,
        hid=128,
        diffusion_steps=diffusion_steps,
        schedule_type='linear'
    ).to(device)
    
    # Load trained weights
    if model_path is None:
        model_path = f"boundary_constrained_madp_{scenario_name}_na_{num_agents}.pth"  # Match v2_2 naming
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded moving horizon model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the v2_2 model first.")
        return None, None
    
    model.eval()
    
    # Get test data
    with torch.no_grad():
        frame, start, goal, na, full_traj = next(iter(test_loader))
        frame = frame.to(device)
        start = start.to(device)
        goal = goal.to(device)
        na = na.to(device)
        
        print(f"Test case: {na.item()} active agents")
        
        # Extract test positions for VMAS
        n_active = na.item()
        test_start_positions = start[0, :n_active].cpu().numpy()
        test_goal_positions = goal[0, :n_active].cpu().numpy()
        
        # Denormalization parameters
        xy_mean = dataset.xy_mean.to(device)
        xy_std = dataset.xy_std.to(device)
        
        # Denormalize start and goal positions
        start_denorm = start * (3 * xy_std) + xy_mean
        goal_denorm = goal * (3 * xy_std) + xy_mean
        
        test_start_positions_denorm = start_denorm[0, :n_active].cpu().numpy()
        test_goal_positions_denorm = goal_denorm[0, :n_active].cpu().numpy()
        print(f"Start positions: {test_start_positions_denorm}")
        print(f"Goal positions: {test_goal_positions_denorm}")
    
    # Initialize moving horizon executor
    vmas_scenario_name = f"{scenario_name}_test"
    print("vmas_scenario_name: ", vmas_scenario_name)
    # vmas_scenario_name =scenario_name
    executor = MovingHorizonTrajectoryExecutor(
        scenario_name=vmas_scenario_name,
        num_envs=1,
        max_steps=full_horizon + 10,  # Allow extra steps
        n_agents=num_agents,
        device=device,
        use_test_positions=True,
        test_start_positions=test_start_positions,
        test_goal_positions=test_goal_positions,
        horizon_size=horizon_size
    )
    
    print("Executing MOVING HORIZON trajectory in VMAS...")

    render = True
    
    # Execute moving horizon trajectory
    results = executor.execute_moving_horizon_trajectory(
        model, frame, start_denorm, goal_denorm, na, full_horizon,
        render=render, store_frames=True
    )
    
    # Create outputs
    if render:
        gif_name = f"madp_{scenario_name}_moving_horizon_validation.gif"
        executor.create_gif_from_frames(results, gif_name)
    
    plot_name = f"madp_{scenario_name}_moving_horizon_plot.png"
    executor.plot_moving_horizon_trajectories(results, plot_name)
    
    # Evaluate performance
    metrics = executor.evaluate_moving_horizon_performance(results)
    
    print("\nMOVING HORIZON Trajectory Execution Results:")
    print(f"Active Agents: {n_active}")
    print(f"Total Horizons Generated: {metrics.get('total_horizons', 'N/A')}")
    print(f"Average Generation Time: {metrics.get('avg_generation_time', 0):.4f}s")
    print(f"Real-time Capable: {metrics.get('real_time_capable', False)}")
    print(f"Mean Tracking Error: {metrics.get('mean_tracking_error', 0):.4f}")
    print(f"Max Tracking Error: {metrics.get('max_tracking_error', 0):.4f}")
    print(f"Final Position Error: {metrics.get('final_position_error', 0):.4f}")
    print(f"Total Reward: {metrics.get('total_reward', 0):.4f}")
    print(f"Success Rate: {metrics.get('success_rate', 0):.2f}")
    
    return results, metrics


# Usage example
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Execute moving horizon trajectory
    results, metrics = execute_moving_horizon_diffusion_trajectory_in_vmas(
        "navigation_v2",
        model_path="boundary_constrained_madp_navigation_v2_na_4.pth",  # v2_2 model
        device=device
    )
    
    if results is not None:
        print(f"\nMoving horizon trajectory execution completed successfully!")
        print(f"GIF and plots created for analysis")
    else:
        print("Execution failed. Please check the configuration and model files.")
