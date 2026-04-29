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
import time

# Import from your modules
from MADP_diffusion_v2_2 import EnhancedMultiAgentDiffusionModel
from MADP_train_and_sample_v2_2 import NormalizedTrajectoryDataset, collate_fn
from vmas.scenarios.navigation_v2_test import HeuristicPolicy
# from vmas.scenarios.navigation_corridor_v2_test import HeuristicPolicy


class ThreeStepHierarchicalController:
    """
    Three-step hierarchical moving horizon controller implementation:
    Step 1: Distance >= 0.5m - MADP trajectory tracking with PD control
    Step 2: 0.1m <= Distance < 0.5m - Greedy goal pursuit with PD control
    Step 3: Distance < 0.1m - Precision control with CLF-QP
    """

    def __init__(self, scenario_name=None, num_envs=1, max_steps=40, n_agents=4,
                 device='cuda', test_start_positions=None, test_goal_positions=None,
                 horizon_size=8, goal_tolerance=0.05, trajectory_threshold=0.5,
                 precision_threshold=0.1):
        """
        Initialize the three-step hierarchical controller
        
        Args:
            scenario_name: VMAS scenario name
            num_envs: Number of environments
            max_steps: Maximum execution steps
            n_agents: Number of agents
            device: Computing device ('cuda' or 'cpu')
            test_start_positions: Fixed start positions for testing
            test_goal_positions: Fixed goal positions for testing
            horizon_size: Size of moving horizon window
            goal_tolerance: Distance threshold for success (0.05m)
            trajectory_threshold: Distance threshold for trajectory tracking (0.5m)
            precision_threshold: Distance threshold for precision control (0.1m)
        """
        self.device = device
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.n_agents = n_agents
        self.max_agents = 10  # Architecture requirement
        self.horizon_size = horizon_size
        self.goal_tolerance = goal_tolerance
        self.trajectory_threshold = trajectory_threshold
        self.precision_threshold = precision_threshold
        self.test_start_positions = test_start_positions
        self.test_goal_positions = test_goal_positions

        # Initialize VMAS environment
        self.env = make_env(
            scenario=scenario_name,
            num_envs=num_envs,
            device=device,
            continuous_actions=True,
            max_steps=max_steps,
            n_agents=n_agents,
            collision_reward=-1.0,
            dist_shaping_factor=1.0,
            final_reward=0.01,
            agent_radius=0.05,
            lidar_range=0.1,
            shared_reward=True,
            use_test_positions=True,
            test_start_positions=self.test_start_positions,
            test_goal_positions=self.test_goal_positions,
            goal_tolerance=self.goal_tolerance
        )

        # Initialize CLF-QP controllers for precision control
        self.heuristic_controllers = []
        for i in range(n_agents):
            controller = HeuristicPolicy(
                continuous_action=True,
                clf_epsilon=0.2,
                clf_slack=10.0
            )
            self.heuristic_controllers.append(controller)
        print(f"Initialized {len(self.heuristic_controllers)} CLF-QP controllers")

        # Agent state tracking
        self.agents_successful = torch.zeros(n_agents, dtype=torch.bool, device=device)
        self.agents_in_trajectory_zone = torch.zeros(n_agents, dtype=torch.bool, device=device)
        self.agents_in_goal_pursuit_zone = torch.zeros(n_agents, dtype=torch.bool, device=device)
        self.agents_in_precision_zone = torch.zeros(n_agents, dtype=torch.bool, device=device)
        self.agents_stopped = torch.zeros(n_agents, dtype=torch.bool, device=device)

        # PD controller parameters
        self.Kp = 10.0  # Base position gain
        self.Kv = 4.0   # Base velocity gain
        self.max_force = 1.0  # Force limit

        # Performance tracking
        self.execution_metrics = {
            'control_modes': [],
            'tracking_errors': [],
            'generation_times': [],
            'rewards': [],
            'clf_failures': 0,
            'total_steps': 0
        }

    def smooth_trajectory_segment(self, trajectory_segment, method='cubic_spline', strength=0.1):
        """
        Smooth trajectory segment for better tracking performance
        
        Args:
            trajectory_segment: Raw trajectory segment [T, n_agents, 2]
            method: Smoothing method ('velocity_smoothing' or 'cubic_spline')
            strength: Smoothing strength (0.0 to 1.0)
        
        Returns:
            Smoothed trajectory segment
        """
        T, Na, D = trajectory_segment.shape
        smoothed = trajectory_segment.clone()
        
        for agent_idx in range(Na):
            agent_traj = trajectory_segment[:, agent_idx].cpu().numpy()
            
            if method == 'velocity_smoothing':
                for dim in range(D):
                    if T > 1:
                        # Compute velocities
                        velocities = np.diff(agent_traj[:, dim])
                        if len(velocities) > 1:
                            # Apply moving average to velocities
                            window_size = min(3, len(velocities))
                            kernel = np.ones(window_size) / window_size
                            smoothed_velocities = np.convolve(velocities, kernel, mode='same')
                            
                            # Reconstruct positions
                            smoothed_positions = np.cumsum(
                                np.concatenate([[agent_traj[0, dim]], smoothed_velocities])
                            )
                            
                            # Blend with original
                            final_positions = (1 - strength) * agent_traj[:, dim] + strength * smoothed_positions
                            
                            # Preserve endpoints
                            final_positions[0] = agent_traj[0, dim]
                            final_positions[-1] = agent_traj[-1, dim]
                            
                            smoothed[:, agent_idx, dim] = torch.tensor(
                                final_positions, device=trajectory_segment.device
                            )
            
            elif method == 'cubic_spline' and T > 3:
                t_points = np.linspace(0, 1, T)
                for dim in range(D):
                    try:
                        cs = CubicSpline(t_points, agent_traj[:, dim], bc_type='natural')
                        smoothed_values = cs(t_points)
                        
                        # Blend with original
                        final_values = (1 - strength) * agent_traj[:, dim] + strength * smoothed_values
                        
                        # Preserve endpoints
                        final_values[0] = agent_traj[0, dim]
                        final_values[-1] = agent_traj[-1, dim]
                        
                        smoothed[:, agent_idx, dim] = torch.tensor(
                            final_values, device=trajectory_segment.device
                        )
                    except:
                        # Fallback to original if spline fails
                        continue
        
        return smoothed

    def get_agent_observations(self):
        """Get agent observations in the format expected by HeuristicPolicy"""
        observations = []
        for agent in self.env.agents:
            obs = self.env.scenario.observation(agent)
            observations.append(obs[0])  # Get first batch element
        return torch.stack(observations)

    def update_agent_status(self, current_positions, goal_positions):
        """
        Update agent status based on distance to goals for three-step control
        
        Args:
            current_positions: Current agent positions [n_agents, 2]
            goal_positions: Goal positions [n_agents, 2]
        
        Returns:
            distances: Distance to goals for each agent
        """
        distances = torch.norm(current_positions - goal_positions, dim=-1)
        
        # Update successful agents (within goal_tolerance)
        newly_successful = distances < self.goal_tolerance
        self.agents_successful = newly_successful
        
        # Update agents in different control zones
        self.agents_in_trajectory_zone = distances >= self.trajectory_threshold
        self.agents_in_goal_pursuit_zone = (distances >= self.precision_threshold) & (distances < self.trajectory_threshold)
        self.agents_in_precision_zone = distances < self.precision_threshold
        
        # Stop successful agents
        self.agents_stopped = self.agents_successful.clone()
        
        return distances

    def compute_three_step_control(self, target_positions, target_velocities, goal_positions):
        """
        Compute three-step hierarchical control actions:
        Step 1: Distance >= 0.5m - MADP trajectory tracking with PD control
        Step 2: 0.1m <= Distance < 0.5m - Greedy goal pursuit with PD control
        Step 3: Distance < 0.1m - Precision control with CLF-QP
        """
        # Get current agent states
        current_positions = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)[0]
        current_velocities = torch.stack([agent.state.vel for agent in self.env.agents], dim=1)[0]
        
        # Ensure tensor compatibility
        target_positions = target_positions.to(self.device)
        target_velocities = target_velocities.to(self.device)
        goal_positions = goal_positions.to(self.device)
        
        # Update agent status
        distances = self.update_agent_status(current_positions, goal_positions)
        
        # Get observations for CLF-QP
        observations = self.get_agent_observations()
        u_range = torch.tensor([self.max_force], device=self.device, dtype=torch.float32)
        
        # Initialize actions
        actions = torch.zeros(self.n_agents, 2, device=self.device, dtype=torch.float32)
        current_modes = []
        
        # Process each agent with three-step control
        for i in range(self.n_agents):
            try:
                if self.agents_stopped[i]:
                    # Agent successfully reached goal
                    actions[i] = torch.zeros(2, device=self.device, dtype=torch.float32)
                    current_modes.append('stopped')
                    
                elif distances[i] >= self.trajectory_threshold:
                    # STEP 1: MADP Trajectory Tracking (Distance >= 0.5m)
                    pos_error = target_positions[i] - current_positions[i]
                    vel_error = target_velocities[i] - current_velocities[i]
                    
                    # Adaptive gains for trajectory tracking
                    kp_adaptive = self.Kp * 1.2  # Slightly higher for trajectory following
                    kv_adaptive = self.Kv * 1.0
                    
                    actions[i] = torch.clamp(
                        kp_adaptive * pos_error + kv_adaptive * vel_error,
                        -self.max_force, self.max_force
                    )
                    current_modes.append('trajectory_tracking')
                    
                elif distances[i] >= self.precision_threshold:
                    # STEP 2: Greedy Goal Pursuit (0.1m <= Distance < 0.5m)
                    pos_error = goal_positions[i] - current_positions[i]
                    vel_error = -current_velocities[i]  # Velocity damping
                    
                    # Adaptive gains for goal pursuit
                    distance_factor = (distances[i] - self.precision_threshold) / (self.trajectory_threshold - self.precision_threshold)
                    kp_adaptive = self.Kp * (0.8 + 0.4 * distance_factor)  # Reduce gain as approaching
                    kv_adaptive = self.Kv * (1.0 + 0.5 * distance_factor)  # Increase damping when closer
                    
                    actions[i] = torch.clamp(
                        kp_adaptive * pos_error + kv_adaptive * vel_error,
                        -self.max_force, self.max_force
                    )
                    current_modes.append('goal_pursuit')
                    
                else:
                    # STEP 3: Precision Control (Distance < 0.1m)
                    success = self._compute_precision_control_action(
                        i, observations, u_range, actions, current_positions,
                        current_velocities, goal_positions
                    )
                    
                    if success:
                        current_modes.append('precision_control')
                    else:
                        current_modes.append('precision_fallback')
                        self.execution_metrics['clf_failures'] += 1
                        
            except Exception as e:
                print(f"Error in three-step control for agent {i}: {e}")
                # Emergency fallback
                pos_error = goal_positions[i] - current_positions[i]
                actions[i] = torch.clamp(pos_error * 1.5, -self.max_force, self.max_force)
                current_modes.append('emergency_fallback')
                self.execution_metrics['clf_failures'] += 1
        
        # Store control modes
        self.execution_metrics['control_modes'].append(current_modes)
        
        return torch.clamp(actions, -self.max_force, self.max_force)

    def _compute_precision_control_action(self, agent_idx, observations, u_range, actions,
                                        current_positions, current_velocities, goal_positions):
        """
        Compute CLF-QP action for high-precision goal achievement
        
        Returns:
            bool: True if CLF-QP succeeded, False if fallback was used
        """
        try:
            # Use CLF-QP for precise control
            obs_cpu = observations[agent_idx].detach().cpu()
            u_range_cpu = u_range.detach().cpu()
            
            if obs_cpu.dim() == 1:
                obs_cpu = obs_cpu.unsqueeze(0)
            
            action_cpu = self.heuristic_controllers[agent_idx].compute_action(
                obs_cpu, u_range_cpu
            )
            
            if action_cpu.dim() > 1:
                action_cpu = action_cpu.squeeze(0)
            
            actions[agent_idx] = action_cpu.to(self.device)
            return True
            
        except Exception as e:
            # Fallback to gentle PD control for precision
            pos_error = goal_positions[agent_idx] - current_positions[agent_idx]
            vel_error = -current_velocities[agent_idx]
            
            # Very gentle gains for precision
            kp_precision = self.Kp * 0.5
            kv_precision = self.Kv * 1.5
            
            actions[agent_idx] = torch.clamp(
                kp_precision * pos_error + kv_precision * vel_error,
                -self.max_force * 0.5, self.max_force * 0.5  # Reduced force limit
            )
            return False

    def generate_trajectory_horizon(self, model, frames, current_positions, goals, n_agents):
        """
        Generate trajectory segment for current horizon using MADP model
        
        Args:
            model: MADP diffusion model
            frames: Environment frames
            current_positions: Current agent positions
            goals: Goal positions
            n_agents: Number of active agents
        
        Returns:
            Smoothed trajectory segment [horizon_size, n_agents, 2]
        """
        start_time = time.time()

        import pdb
        pdb.set_trace()
        
        with torch.no_grad():
            # Generate trajectory using diffusion model
            horizon_prediction = model.sample_with_constraints(
                frames, current_positions, goals, n_agents,
                steps=50,  # Diffusion sampling steps
                max_step_size=0.1
            )
        
        generation_time = time.time() - start_time
        self.execution_metrics['generation_times'].append(generation_time)
        
        # Extract active agents and reformat
        n_active = n_agents.item()
        horizon_segment = horizon_prediction[0, :n_active].permute(1, 0, 2)
        
        # Smooth the trajectory segment
        smoothed_segment = self.smooth_trajectory_segment(
            horizon_segment, method='cubic_spline', strength=0.1
        )
        
        return smoothed_segment, generation_time

    def execute_three_step_trajectory(self, model, frames, starts, goals, n_agents,
                                    full_horizon_length, render=True, store_frames=True):
        """
        Execute complete three-step hierarchical moving horizon trajectory
        
        This is the main execution loop that combines:
        1. Moving horizon trajectory generation
        2. Three-step hierarchical control
        3. Performance monitoring and visualization
        
        Args:
            model: Trained MADP diffusion model
            frames: Environment frames
            starts: Start positions
            goals: Goal positions
            n_agents: Number of active agents
            full_horizon_length: Total trajectory length
            render: Whether to render
            store_frames: Whether to store frames for GIF
        
        Returns:
            Dictionary with execution results and metrics
        """
        print(f"Starting THREE-STEP HIERARCHICAL MOVING HORIZON execution")
        print(f"Step 1 - Trajectory Tracking: Distance >= {self.trajectory_threshold}m")
        print(f"Step 2 - Goal Pursuit: {self.precision_threshold}m <= Distance < {self.trajectory_threshold}m")
        print(f"Step 3 - Precision Control: Distance < {self.precision_threshold}m")
        print(f"Goal tolerance: {self.goal_tolerance}")
        print(f"Horizon size: {self.horizon_size}")
        print(f"Active agents: {n_agents.item()}")
        
        # Reset environment
        obs = self.env.reset()
        
        # Initialize storage
        executed_positions = []
        planned_segments = []
        
        # Initialize frames storage
        if store_frames:
            self.env.frames = []
        
        # Reset execution metrics
        self.execution_metrics = {
            'control_modes': [],
            'tracking_errors': [],
            'generation_times': [],
            'rewards': [],
            'clf_failures': 0,
            'total_steps': 0
        }
        
        # Initialize agent positions
        current_positions = starts.clone()
        
        # Reset agent status
        n_active = n_agents.item()
        self.agents_successful = torch.zeros(n_active, dtype=torch.bool, device=self.device)
        self.agents_in_trajectory_zone = torch.zeros(n_active, dtype=torch.bool, device=self.device)
        self.agents_in_goal_pursuit_zone = torch.zeros(n_active, dtype=torch.bool, device=self.device)
        self.agents_in_precision_zone = torch.zeros(n_active, dtype=torch.bool, device=self.device)
        self.agents_stopped = torch.zeros(n_active, dtype=torch.bool, device=self.device)
        
        # Extract goal positions
        goal_positions = goals[0, :n_active]
        
        # Current trajectory segment
        current_segment = None
        segment_start_step = 0
        
        # Main execution loop
        for step in range(self.max_steps):
            self.execution_metrics['total_steps'] = step + 1
            
            # Generate new horizon if needed
            if step % self.horizon_size == 0:
                print(f"Generating new horizon at step {step}")
                
                # Generate trajectory segment
                current_segment, generation_time = self.generate_trajectory_horizon(
                    model, frames, current_positions, goals, n_agents
                )
                
                planned_segments.append(current_segment.cpu())
                segment_start_step = step
                print(f"Horizon generated in {generation_time:.3f}s")
            
            # Get target positions and velocities from current segment
            segment_idx = step - segment_start_step
            if current_segment is not None and segment_idx < len(current_segment) - 1:
                # Use next position in segment as target
                target_positions = current_segment[segment_idx + 1, :n_active]
                
                # Compute target velocities
                if segment_idx < len(current_segment) - 2:
                    target_velocities = (
                        current_segment[segment_idx + 2, :n_active] - 
                        current_segment[segment_idx, :n_active]
                    ) / 2.0
                else:
                    target_velocities = torch.zeros_like(target_positions)
            else:
                # Use goals as targets if no segment available
                target_positions = goal_positions
                target_velocities = torch.zeros_like(target_positions)
            
            # Compute three-step hierarchical control actions
            control_actions = self.compute_three_step_control(
                target_positions, target_velocities, goal_positions
            )
            
            # Convert to VMAS action format
            actions = []
            for i in range(self.n_agents):
                if i < n_active:
                    actions.append(control_actions[i].unsqueeze(0))
                else:
                    actions.append(torch.zeros(1, 2, device=self.device))
            
            # Step environment
            obs, rewards, dones, info = self.env.step(actions)
            
            # Rendering
            if store_frames and render:
                self.rendering_callback(self.env)
            
            # Record current positions
            current_env_positions = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)
            executed_positions.append(current_env_positions[0, :n_active].cpu())
            
            # Compute tracking error
            if current_segment is not None and segment_idx < len(current_segment):
                expected_pos = current_segment[segment_idx, :n_active].cpu()
                actual_pos = current_env_positions[0, :n_active].cpu()
                tracking_error = torch.norm(actual_pos - expected_pos, dim=-1).mean()
                self.execution_metrics['tracking_errors'].append(tracking_error.item())
            
            # Record rewards
            total_reward = sum([r.sum().item() for r in rewards])
            self.execution_metrics['rewards'].append(total_reward)
            
            # Check termination conditions
            if dones.any():
                print(f"Episode finished at step {step}")
                break
            
            if self.agents_successful.all():
                print(f"All agents reached their goals at step {step}")
                break
            
            # Update current positions for next model input
            current_positions = torch.zeros_like(starts)
            current_positions[0, :n_active] = current_env_positions[0, :n_active]
        
        # Convert results to tensors
        executed_positions = torch.stack(executed_positions) if executed_positions else torch.empty(0, n_active, 2)
        
        # Print execution summary
        print(f"Execution completed:")
        print(f" Generated {len(planned_segments)} horizon segments")
        print(f" Average generation time: {np.mean(self.execution_metrics['generation_times']):.3f}s")
        print(f" CLF-QP failures: {self.execution_metrics['clf_failures']}")
        print(f" Final success rate: {self.agents_successful.float().mean().item():.2f}")
        
        return {
            'executed_positions': executed_positions,
            'planned_segments': planned_segments,
            'control_modes': self.execution_metrics['control_modes'],
            'tracking_errors': self.execution_metrics['tracking_errors'],
            'rewards': self.execution_metrics['rewards'],
            'frames': self.env.frames if hasattr(self.env, 'frames') else None,
            'generation_times': self.execution_metrics['generation_times'],
            'num_horizons': len(planned_segments),
            'agents_successful': self.agents_successful.cpu(),
            'final_distances': self.compute_final_distances(),
            'clf_failures': self.execution_metrics['clf_failures'],
            'total_steps': self.execution_metrics['total_steps']
        }

    def compute_final_distances(self):
        """Compute final distances to goals for all agents"""
        current_positions = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)
        goal_positions = torch.tensor(self.test_goal_positions, device=self.device)
        distances = torch.norm(current_positions[0, :self.n_agents] - goal_positions, dim=-1)
        return distances.cpu()

    def rendering_callback(self, env):
        """VMAS rendering callback"""
        frame = env.render(mode="rgb_array", agent_index_focus=None, visualize_when_rgb=True)
        if frame is not None:
            env.frames.append(Image.fromarray(frame))

    def create_gif_from_frames(self, results, gif_name="three_step_execution.gif"):
        """Create GIF from stored frames"""
        if results['frames'] and len(results['frames']) > 0:
            results['frames'][0].save(
                gif_name,
                save_all=True,
                append_images=results['frames'][1:],
                duration=100,
                loop=1
            )
            print(f"GIF saved as {gif_name}")
            return True
        else:
            print("No frames to save")
            return False

    def evaluate_performance(self, results):
        """Comprehensive performance evaluation for three-step control"""
        executed = results['executed_positions']
        if len(executed) == 0:
            return {'error': 'No execution data available'}
        
        # Success metrics
        final_distances = results['final_distances']
        agents_successful = final_distances < self.goal_tolerance
        success_rate = agents_successful.float().mean().item()
        
        # Control mode analysis for three-step control
        control_modes = results['control_modes']
        mode_counts = {
            'trajectory_tracking': 0,  # Step 1: MADP trajectory following
            'goal_pursuit': 0,         # Step 2: Greedy goal pursuit
            'precision_control': 0,    # Step 3: CLF-QP precision control
            'precision_fallback': 0,   # Step 3 fallback
            'stopped': 0,              # Successful agents
            'emergency_fallback': 0    # Emergency fallback
        }
        
        for step_modes in control_modes:
            for mode in step_modes:
                if mode in mode_counts:
                    mode_counts[mode] += 1
        
        # Tracking performance
        tracking_errors = results['tracking_errors']
        mean_tracking_error = np.mean(tracking_errors) if tracking_errors else 0.0
        max_tracking_error = np.max(tracking_errors) if tracking_errors else 0.0
        
        # Generation performance
        generation_times = results['generation_times']
        avg_generation_time = np.mean(generation_times) if generation_times else 0.0
        max_generation_time = np.max(generation_times) if generation_times else 0.0
        
        # Overall performance
        total_reward = sum(results['rewards'])
        clf_failure_rate = results['clf_failures'] / results['total_steps'] if results['total_steps'] > 0 else 0.0
        
        metrics = {
            'success_rate': success_rate,
            'final_distances': final_distances.tolist(),
            'agents_successful': agents_successful.tolist(),
            'mean_tracking_error': mean_tracking_error,
            'max_tracking_error': max_tracking_error,
            'avg_generation_time': avg_generation_time,
            'max_generation_time': max_generation_time,
            'total_reward': total_reward,
            'control_mode_counts': mode_counts,
            'clf_failure_rate': clf_failure_rate,
            'real_time_capable': avg_generation_time < 0.1,
            'num_horizons': results['num_horizons'],
            'total_steps': results['total_steps']
        }
        
        return metrics

    def plot_comprehensive_analysis(self, results, plot_name="three_step_analysis.png"):
        """Create comprehensive analysis plots for three-step control"""
        executed = results['executed_positions']
        if len(executed) == 0:
            print("No execution data to plot")
            return None
        
        # Reconstruct full planned trajectory
        planned_segments = results['planned_segments']
        if len(planned_segments) > 0:
            planned_full = torch.cat([seg[:-1] for seg in planned_segments[:-1]] + [planned_segments[-1]], dim=0)
            min_length = min(len(executed), len(planned_full))
            executed = executed[:min_length]
            planned_full = planned_full[:min_length]
        
        plt.figure(figsize=(20, 16))
        
        # 1. Trajectory Comparison
        plt.subplot(2, 3, 1)
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i in range(executed.shape[1]):
            plt.plot(executed[:, i, 0], executed[:, i, 1], 'o-',
                    color=colors[i % len(colors)], label=f'Agent {i} Executed', markersize=2)
            if len(planned_segments) > 0:
                plt.plot(planned_full[:, i, 0], planned_full[:, i, 1], 'x--',
                        color=colors[i % len(colors)], label=f'Agent {i} Planned', alpha=0.7)
        
        plt.title('Three-Step Hierarchical Trajectory Comparison')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # 2. Control Mode Timeline
        plt.subplot(2, 3, 2)
        control_modes = results['control_modes']
        mode_colors = {
            'trajectory_tracking': 'blue',     # Step 1
            'goal_pursuit': 'green',           # Step 2
            'precision_control': 'orange',     # Step 3
            'precision_fallback': 'red',       # Step 3 fallback
            'stopped': 'purple',               # Stopped agents
            'emergency_fallback': 'black'      # Emergency fallback
        }
        
        for i in range(len(control_modes[0]) if control_modes else 0):
            mode_timeline = [modes[i] for modes in control_modes]
            for j, mode in enumerate(mode_timeline):
                if mode in mode_colors:
                    plt.scatter(j, i, c=mode_colors[mode], s=15, alpha=0.8)
        
        plt.title('Three-Step Control Mode Timeline')
        plt.xlabel('Time Step')
        plt.ylabel('Agent Index')
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color, markersize=10, label=mode)
                          for mode, color in mode_colors.items()]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 3. Tracking Error
        plt.subplot(2, 3, 3)
        tracking_errors = results['tracking_errors']
        if tracking_errors:
            plt.plot(tracking_errors, 'r-', linewidth=2)
            plt.axhline(y=np.mean(tracking_errors), color='g', linestyle='--',
                       label=f'Mean: {np.mean(tracking_errors):.3f}')
            plt.title('Tracking Error Over Time')
            plt.xlabel('Time Step')
            plt.ylabel('Tracking Error (m)')
            plt.legend()
            plt.grid(True)
        
        # 4. Generation Times
        plt.subplot(2, 3, 4)
        generation_times = results['generation_times']
        if generation_times:
            plt.bar(range(len(generation_times)), generation_times)
            plt.axhline(y=0.1, color='r', linestyle='--', label='Real-time threshold')
            plt.axhline(y=np.mean(generation_times), color='g', linestyle='--',
                       label=f'Average: {np.mean(generation_times):.3f}s')
            plt.title('Generation Time per Horizon')
            plt.xlabel('Horizon Index')
            plt.ylabel('Time (s)')
            plt.legend()
            plt.grid(True)
        
        # 5. Cumulative Rewards
        plt.subplot(2, 3, 5)
        rewards = results['rewards']
        if rewards:
            cumulative_rewards = np.cumsum(rewards)
            plt.plot(rewards, 'b-', alpha=0.7, label='Step Reward')
            plt.plot(cumulative_rewards, 'g-', linewidth=2, label='Cumulative Reward')
            plt.title('Reward Progress')
            plt.xlabel('Time Step')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)
        
        # 6. Final Performance
        plt.subplot(2, 3, 6)
        final_distances = results['final_distances']
        agents_successful = final_distances < self.goal_tolerance
        bar_colors = ['green' if success else 'red' for success in agents_successful]
        plt.bar(range(len(final_distances)), final_distances, color=bar_colors)
        plt.axhline(y=self.goal_tolerance, color='black', linestyle='--',
                   label=f'Goal Tolerance: {self.goal_tolerance}')
        plt.axhline(y=self.precision_threshold, color='orange', linestyle='--',
                   label=f'Precision Threshold: {self.precision_threshold}')
        plt.axhline(y=self.trajectory_threshold, color='blue', linestyle='--',
                   label=f'Trajectory Threshold: {self.trajectory_threshold}')
        plt.title('Final Distance to Goals')
        plt.xlabel('Agent Index')
        plt.ylabel('Distance (m)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plot_name, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_name


def execute_three_step_hierarchical_system(scenario_name, model_path=None, device='cuda',
                                         goal_tolerance=0.05, trajectory_threshold=0.5,
                                         precision_threshold=0.1, horizon_size=8):
    """
    Main execution function for the complete three-step hierarchical moving horizon system
    
    Args:
        scenario_name: VMAS scenario name
        model_path: Path to trained model
        device: Computing device
        goal_tolerance: Success threshold (0.05)
        trajectory_threshold: Threshold for trajectory tracking (0.5)
        precision_threshold: Threshold for precision control (0.1)
        horizon_size: Moving horizon window size (8)
    
    Returns:
        Tuple of (results, metrics)
    """
    # Load training configuration
    train_config_file = "MADP_training_config.yaml"
    try:
        with open(train_config_file, 'r') as file:
            train_config = yaml.safe_load(file)
        scenario_name = train_config['param']['scenario']
        num_agents = train_config['param']['num_agents']
        full_horizon = train_config['param']['horizon']
        h5_path = f"{scenario_name}_Na_{num_agents}_T_{full_horizon}_dataset.h5"
        diffusion_steps = train_config['param']['diffuse_steps']
    except FileNotFoundError:
        print("Warning: Training config not found, using defaults")
        scenario_name = "navigation_v3"
        num_agents = 4
        full_horizon = 40
        h5_path = "dataset.h5"
        diffusion_steps = 500
    
    print(f"Initializing Three-Step Hierarchical Moving Horizon System")
    print(f"Scenario: {scenario_name}")
    print(f"Active Agents: {num_agents}")
    print(f"Full Horizon: {full_horizon}")
    print(f"Moving Horizon Size: {horizon_size}")
    print(f"Goal Tolerance: {goal_tolerance}")
    print(f"Trajectory Threshold: {trajectory_threshold}")
    print(f"Precision Threshold: {precision_threshold}")
    
    # Load dataset
    try:
        dataset = NormalizedTrajectoryDataset(h5_path, split='test', horizon=full_horizon)
        test_loader = DataLoader(dataset, 1, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None
    
    # Load model
    model = EnhancedMultiAgentDiffusionModel(
        max_agents=10,
        horizon=horizon_size,
        state_dim=2,
        img_ch=3,
        hid=128,
        diffusion_steps=diffusion_steps,
        schedule_type='linear'
    ).to(device)
    
    if model_path is None:
        model_path = f"boundary_constrained_madp_{scenario_name}.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found")
        return None, None
    
    model.eval()
    
    # Get test data
    try:
        with torch.no_grad():
            frame, start, goal, na, full_traj = next(iter(test_loader))
            frame = frame.to(device)
            start = start.to(device)
            goal = goal.to(device)
            na = na.to(device)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None
    
    # Denormalize positions
    xy_mean = dataset.xy_mean.to(device)
    xy_std = dataset.xy_std.to(device)
    start_denorm = start * (3 * xy_std) + xy_mean
    goal_denorm = goal * (3 * xy_std) + xy_mean
    
    n_active = na.item()
    test_start_positions = start_denorm[0, :n_active].cpu().numpy()
    test_goal_positions = goal_denorm[0, :n_active].cpu().numpy()
    
    print(f"Test case loaded:")
    print(f" Active agents: {n_active}")
    print(f" Start positions: {test_start_positions}")
    print(f" Goal positions: {test_goal_positions}")
    
    # Initialize three-step hierarchical controller
    vmas_scenario_name = f"{scenario_name}_test"
    controller = ThreeStepHierarchicalController(
        scenario_name=vmas_scenario_name,
        num_envs=1,
        max_steps=full_horizon + 50,
        n_agents=num_agents,
        device=device,
        test_start_positions=test_start_positions,
        test_goal_positions=test_goal_positions,
        horizon_size=horizon_size,
        goal_tolerance=goal_tolerance,
        trajectory_threshold=trajectory_threshold,
        precision_threshold=precision_threshold
    )
    
    # Execute three-step hierarchical trajectory
    print("\nExecuting three-step hierarchical moving horizon trajectory...")
    results = controller.execute_three_step_trajectory(
        model, frame, start_denorm, goal_denorm, na, full_horizon,
        render=True, store_frames=True
    )
    
    # Create visualizations
    gif_name = f"three_step_hierarchical_madp_{scenario_name}_na_{num_agents}.gif"
    controller.create_gif_from_frames(results, gif_name)
    
    plot_name = f"three_step_hierarchical_analysis_{scenario_name}.png"
    controller.plot_comprehensive_analysis(results, plot_name)
    
    # Evaluate performance
    metrics = controller.evaluate_performance(results)
    
    # Print comprehensive results
    print("\n" + "="*70)
    print("THREE-STEP HIERARCHICAL MOVING HORIZON SYSTEM - EXECUTION RESULTS")
    print("="*70)
    print(f"Success Rate: {metrics['success_rate']:.2f}")
    print(f"Mean Tracking Error: {metrics['mean_tracking_error']:.4f}")
    print(f"Max Tracking Error: {metrics['max_tracking_error']:.4f}")
    print(f"Average Generation Time: {metrics['avg_generation_time']:.4f}s")
    print(f"Max Generation Time: {metrics['max_generation_time']:.4f}s")
    print(f"Real-time Capable: {metrics['real_time_capable']}")
    print(f"Total Reward: {metrics['total_reward']:.4f}")
    print(f"CLF-QP Failure Rate: {metrics['clf_failure_rate']:.4f}")
    print(f"Total Steps: {metrics['total_steps']}")
    print(f"Horizons Generated: {metrics['num_horizons']}")
    
    print(f"\nThree-Step Control Mode Distribution:")
    for mode, count in metrics['control_mode_counts'].items():
        percentage = (count / sum(metrics['control_mode_counts'].values())) * 100
        print(f" {mode}: {count} steps ({percentage:.1f}%)")
    
    print(f"\nPer-Agent Results:")
    for i, (distance, success) in enumerate(zip(metrics['final_distances'], metrics['agents_successful'])):
        status = "SUCCESS" if success else "FAILED"
        print(f" Agent {i}: {distance:.3f}m to goal [{status}]")
    
    print(f"\nOutput Files:")
    print(f" GIF: {gif_name}")
    print(f" Analysis Plot: {plot_name}")
    
    return results, metrics


# Main execution
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    final_succ_rate = []
    for _ in range(10):
        # Execute the three-step hierarchical system
        results, metrics = execute_three_step_hierarchical_system(
            scenario_name="navigation_v2",
            model_path="boundary_constrained_madp_navigation_v2_na_4_full.pth",
            device=device,
            goal_tolerance=0.1,
            trajectory_threshold=0.75,
            precision_threshold=0.1,
            horizon_size=50
        )
        
        final_succ_rate.append(metrics['success_rate'])
    
    print(final_succ_rate)
    print("Average Success rate: ", np.mean(np.array(final_succ_rate)))
    
    if results is not None:
        print("\n🎉 Three-Step Hierarchical Moving Horizon System executed successfully!")
        print("📊 Check the generated GIF and analysis plots for detailed results.")
    else:
        print("❌ Execution failed. Please check the configuration and model files.")
