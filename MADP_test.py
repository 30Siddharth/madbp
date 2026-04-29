import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image
import time
import h5py
from tqdm import tqdm

# Import from your modules
from MADP_diffusion_v2_2 import EnhancedMultiAgentDiffusionModel
from MADP_train_and_sample_v2_2 import NormalizedTrajectoryDataset, collate_fn
from vmas import make_env
from vmas.scenarios.navigation_v2_test import HeuristicPolicy

class DatasetReader:
    """
    Reader class for HDF5 dataset created by the collection script
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = None
        self.metadata = None
        
    def load_dataset(self):
        """Load the HDF5 dataset and extract metadata"""
        try:
            self.dataset = h5py.File(self.dataset_path, 'r')
            
            # Extract metadata
            self.metadata = {
                'num_agents': self.dataset.attrs['num_agents'],
                'scenario_name': self.dataset.attrs['scenario_name'],
                'max_steps': self.dataset.attrs['max_steps'],
                'num_cases': self.dataset.attrs['num_cases']
            }
            
            print(f"Dataset loaded successfully:")
            print(f"  Scenario: {self.metadata['scenario_name']}")
            print(f"  Number of agents: {self.metadata['num_agents']}")
            print(f"  Max steps: {self.metadata['max_steps']}")
            print(f"  Number of test cases: {self.metadata['num_cases']}")
            
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def get_test_case(self, case_idx):
        """Get a specific test case from the dataset"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        episode_name = f'episode_{case_idx}'
        
        try:
            # Load frame data
            frame_data = self.dataset['frames'][episode_name][:]
            frame_tensor = torch.tensor(frame_data, dtype=torch.float32)
            
            # Load start and goal poses
            start_poses = self.dataset['start_poses'][episode_name][:]
            goal_poses = self.dataset['goal_poses'][episode_name][:]
            
            return {
                'frame': frame_tensor,
                'start_poses': start_poses,
                'goal_poses': goal_poses,
                'case_idx': case_idx
            }
            
        except KeyError:
            raise ValueError(f"Test case {case_idx} not found in dataset")
    
    def close(self):
        """Close the dataset file"""
        if self.dataset:
            self.dataset.close()

class HierarchicalMovingHorizonController:
    """
    Hierarchical controller based on MADP_validate.py implementation
    """
    
    def __init__(self, scenario_name=None, num_envs=1, max_steps=40, n_agents=4,
                 device='cuda', test_start_positions=None, test_goal_positions=None,
                 horizon_size=8, goal_tolerance=0.1, goal_pursuit_threshold=0.25):
        
        self.device = device
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.n_agents = n_agents
        self.max_agents = 10
        self.horizon_size = horizon_size
        self.goal_tolerance = goal_tolerance
        self.goal_pursuit_threshold = goal_pursuit_threshold
        self.test_start_positions = test_start_positions
        self.test_goal_positions = test_goal_positions
        
        # Initialize VMAS environment - CORRECTED based on MADP_validate.py
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
            x_semidim=1.0,
            y_semidim=1.0,
            lidar_range=0.1,
            shared_reward=True,
            use_test_positions=True,  # CORRECTED: Set to True
            test_start_positions=self.test_start_positions,
            test_goal_positions=self.test_goal_positions,
            goal_tolerance=self.goal_tolerance
        )
        
        # Initialize HeuristicPolicy controllers - CORRECTED format
        self.heuristic_controllers = []
        for i in range(n_agents):
            controller = HeuristicPolicy(
                continuous_action=True,  # CORRECTED: Fixed syntax
                clf_epsilon=0.2,
                clf_slack=100.0
            )
            self.heuristic_controllers.append(controller)
        
        print(f"Initialized {len(self.heuristic_controllers)} HeuristicPolicy CLF-QP controllers")
        
        # Agent state tracking
        self.agents_successful = torch.zeros(n_agents, dtype=torch.bool, device=device)
        self.agents_in_goal_pursuit = torch.zeros(n_agents, dtype=torch.bool, device=device)
        self.agents_stopped = torch.zeros(n_agents, dtype=torch.bool, device=device)
        
        # Controller parameters
        self.Kp = 10.0
        self.Kv = 4.0
        self.max_force = 1.0
        
        # Performance tracking
        self.execution_metrics = {
            'control_modes': [],
            'tracking_errors': [],
            'generation_times': [],
            'rewards': [],
            'clf_failures': 0,
            'total_steps': 0
        }
    
    def smooth_trajectory_segment(self, trajectory_segment, method='velocity_smoothing', strength=0.3):
        """Smooth trajectory segment - copied from MADP_validate.py"""
        T, Na, D = trajectory_segment.shape
        smoothed = trajectory_segment.clone()
        
        for agent_idx in range(Na):
            agent_traj = trajectory_segment[:, agent_idx].cpu().numpy()
            
            if method == 'velocity_smoothing':
                for dim in range(D):
                    if T > 1:
                        velocities = np.diff(agent_traj[:, dim])
                        if len(velocities) > 1:
                            window_size = min(3, len(velocities))
                            kernel = np.ones(window_size) / window_size
                            smoothed_velocities = np.convolve(velocities, kernel, mode='same')
                            
                            smoothed_positions = np.cumsum(
                                np.concatenate([[agent_traj[0, dim]], smoothed_velocities])
                            )
                            
                            final_positions = (1 - strength) * agent_traj[:, dim] + strength * smoothed_positions
                            final_positions[0] = agent_traj[0, dim]
                            final_positions[-1] = agent_traj[-1, dim]
                            
                            smoothed[:, agent_idx, dim] = torch.tensor(
                                final_positions, device=trajectory_segment.device
                            )
        
        return smoothed
    
    def get_agent_observations(self):
        """Get agent observations - copied from MADP_validate.py"""
        observations = []
        for agent in self.env.agents:
            obs = self.env.scenario.observation(agent)
            observations.append(obs[0])
        return torch.stack(observations)
    
    def update_agent_status(self, current_positions, goal_positions):
        """Update agent status - copied from MADP_validate.py"""
        distances = torch.norm(current_positions - goal_positions, dim=-1)
        
        newly_successful = distances < self.goal_tolerance
        self.agents_successful = newly_successful
        
        in_pursuit = distances < self.goal_pursuit_threshold
        self.agents_in_goal_pursuit = in_pursuit
        
        self.agents_stopped = self.agents_successful.clone()
        
        return distances
    
    def compute_hierarchical_control(self, target_positions, target_velocities, goal_positions):
        """
        Compute hierarchical control - CORRECTED based on MADP_validate.py
        """
        # Get current agent states
        current_positions = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)[0]
        current_velocities = torch.stack([agent.state.vel for agent in self.env.agents], dim=1)[0]
        
        # Update agent status
        distances = self.update_agent_status(current_positions, goal_positions)
        
        # Get agent observations
        observations = self.get_agent_observations()
        
        # Control range - CORRECTED format
        u_range = torch.tensor([self.max_force], device=self.device)
        
        # Initialize actions
        actions = torch.zeros(self.n_agents, 2, device=self.device)
        
        # Record current control modes
        current_modes = []
        
        for i in range(self.n_agents):
            if self.agents_stopped[i]:
                # Agent has reached goal
                actions[i] = torch.zeros(2, device=self.device)
                current_modes.append('stopped')
                
            elif self.agents_in_goal_pursuit[i]:
                # Agent is close to goal - use CLF-QP for goal pursuit
                current_modes.append('goal_pursuit')
                
                try:
                    # CORRECTED: CPU conversion for CLF-QP
                    obs_cpu = observations[i].detach().cpu().unsqueeze(0)
                    u_range_cpu = u_range.detach().cpu()
                    
                    action = self.heuristic_controllers[i].compute_action(obs_cpu, u_range_cpu)
                    actions[i] = action[0].to(self.device)
                    
                except Exception as e:
                    print(f"CLF-QP goal pursuit failed for agent {i}: {e}")
                    self.execution_metrics['clf_failures'] += 1
                    
                    # Fallback to proportional control
                    pos_error = goal_positions[i] - current_positions[i]
                    vel_error = -current_velocities[i]
                    actions[i] = torch.clamp(
                        self.Kp * pos_error + self.Kv * vel_error,
                        -self.max_force, self.max_force
                    )
            
            else:
                # Agent follows MADP trajectory
                current_modes.append('trajectory_tracking')
                
                try:
                    # CORRECTED: Proper observation modification
                    modified_obs = observations[i].clone().detach()
                    current_pos = current_positions[i]
                    target_relative_pos = current_pos - target_positions[i]
                    
                    # Update goal relative position
                    if modified_obs.shape[0] >= 6:
                        modified_obs[4:6] = target_relative_pos
                    
                    # CPU conversion for CLF-QP
                    obs_cpu = modified_obs.detach().cpu().unsqueeze(0)
                    u_range_cpu = u_range.detach().cpu()
                    
                    action = self.heuristic_controllers[i].compute_action(obs_cpu, u_range_cpu)
                    actions[i] = action[0].to(self.device)
                    
                except Exception as e:
                    print(f"CLF-QP trajectory tracking failed for agent {i}: {e}")
                    self.execution_metrics['clf_failures'] += 1
                    
                    # Fallback to PD control
                    pos_error = target_positions[i] - current_positions[i]
                    vel_error = target_velocities[i] - current_velocities[i]
                    actions[i] = torch.clamp(
                        self.Kp * pos_error + self.Kv * vel_error,
                        -self.max_force, self.max_force
                    )
        
        # Store control modes
        self.execution_metrics['control_modes'].append(current_modes)
        
        return actions
    
    def generate_trajectory_horizon(self, model, frames, current_positions, goals, n_agents):
        """Generate trajectory horizon - copied from MADP_validate.py"""
        start_time = time.time()
        
        with torch.no_grad():
            horizon_prediction = model.sample_with_constraints(
                frames, current_positions, goals, n_agents,
                steps=50,
                max_step_size=0.05
            )
        
        generation_time = time.time() - start_time
        self.execution_metrics['generation_times'].append(generation_time)
        
        n_active = n_agents.item()
        horizon_segment = horizon_prediction[0, :n_active].permute(1, 0, 2)
        
        smoothed_segment = self.smooth_trajectory_segment(
            horizon_segment, method='velocity_smoothing', strength=0.3
        )
        
        return smoothed_segment, generation_time
    
    def execute_hierarchical_trajectory(self, model, frames, starts, goals, n_agents,
                                      full_horizon_length, render=True, store_frames=True):
        """
        Execute hierarchical trajectory - CORRECTED based on MADP_validate.py
        """
        print(f"Starting HIERARCHICAL MOVING HORIZON execution")
        print(f"Goal tolerance: {self.goal_tolerance}")
        print(f"Goal pursuit threshold: {self.goal_pursuit_threshold}")
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
        self.agents_in_goal_pursuit = torch.zeros(n_active, dtype=torch.bool, device=self.device)
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
                import pdb
                pdb.set_trace()
                
                current_segment, generation_time = self.generate_trajectory_horizon(
                    model, frames, current_positions, goals, n_agents
                )
                
                planned_segments.append(current_segment.cpu())
                segment_start_step = step
                
                print(f"Horizon generated in {generation_time:.3f}s")
            
            # Get target positions from current segment
            segment_idx = step - segment_start_step
            
            if current_segment is not None and segment_idx < len(current_segment) - 1:
                target_positions = current_segment[segment_idx + 1, :n_active]
                
                if segment_idx < len(current_segment) - 2:
                    target_velocities = (
                        current_segment[segment_idx + 2, :n_active] - 
                        current_segment[segment_idx, :n_active]
                    ) / 2.0
                else:
                    target_velocities = torch.zeros_like(target_positions)
            else:
                target_positions = goal_positions
                target_velocities = torch.zeros_like(target_positions)
            
            # Compute hierarchical control
            control_actions = self.compute_hierarchical_control(
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
            
            # Update current positions
            current_positions = torch.zeros_like(starts)
            current_positions[0, :n_active] = current_env_positions[0, :n_active]
        
        # Convert results to tensors
        executed_positions = torch.stack(executed_positions) if executed_positions else torch.empty(0, n_active, 2)
        
        # Print execution summary
        print(f"Execution completed:")
        print(f"  Generated {len(planned_segments)} horizon segments")
        print(f"  Average generation time: {np.mean(self.execution_metrics['generation_times']):.3f}s")
        print(f"  CLF-QP failures: {self.execution_metrics['clf_failures']}")
        print(f"  Final success rate: {self.agents_successful.float().mean().item():.2f}")
        
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
        """Compute final distances - copied from MADP_validate.py"""
        current_positions = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)
        goal_positions = torch.tensor(self.test_goal_positions, device=self.device)
        
        distances = torch.norm(current_positions[0, :self.n_agents] - goal_positions, dim=-1)
        return distances.cpu()
    
    def rendering_callback(self, env):
        """Rendering callback - copied from MADP_validate.py"""
        frame = env.render(mode="rgb_array", agent_index_focus=None, visualize_when_rgb=True)
        if frame is not None:
            env.frames.append(Image.fromarray(frame))
    
    def create_gif_from_frames(self, results, gif_name="hierarchical_execution.gif"):
        """Create GIF - copied from MADP_validate.py"""
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

def execute_hierarchical_moving_horizon_system(dataset_path, model_path=None, device='cuda',
                                             goal_tolerance=0.1, goal_pursuit_threshold=0.25,
                                             horizon_size=8, num_test_cases=10):
    """
    Execute hierarchical moving horizon system using dataset
    """
    # Load configuration
    try:
        with open("MADP_training_config.yaml", 'r') as file:
            train_config = yaml.safe_load(file)
        config = train_config['param']
    except FileNotFoundError:
        print("Warning: Config file not found, using defaults")
        config = {
            'scenario': 'navigation_v2',
            'num_agents': 4,
            'horizon': 50,
            'diffuse_steps': 150
        }
    
    print(f"Initializing Hierarchical Moving Horizon System")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    
    # Load dataset
    dataset_reader = DatasetReader(dataset_path)
    if not dataset_reader.load_dataset():
        return None, None
    
    # Load model
    model = EnhancedMultiAgentDiffusionModel(
        max_agents=10,
        horizon=horizon_size,
        state_dim=2,
        img_ch=3,
        hid=128,
        diffusion_steps=config['diffuse_steps'],
        schedule_type='linear'
    ).to(device)
    
    if model_path is None:
        model_path = f"boundary_constrained_madp_{config['scenario']}_na_{config['num_agents']}.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found")
        return None, None
    
    model.eval()
    
    # Run validation on test cases
    all_results = []
    
    for case_idx in range(min(num_test_cases, dataset_reader.metadata['num_cases'])):
        print(f"\nTesting case {case_idx + 1}/{num_test_cases}")
        
        try:
            # Get test case
            test_case = dataset_reader.get_test_case(case_idx)
            
            # Initialize controller
            controller = HierarchicalMovingHorizonController(
                scenario_name="navigation_v2",
                num_envs=1,
                max_steps=config['horizon'] + 20,
                n_agents=config['num_agents'],
                device=device,
                test_start_positions=test_case['start_poses'],
                test_goal_positions=test_case['goal_poses'],
                horizon_size=horizon_size,
                goal_tolerance=goal_tolerance,
                goal_pursuit_threshold=goal_pursuit_threshold
            )
            
            # Format data for execution
            frame = test_case['frame'].to(device)
            starts = torch.tensor(test_case['start_poses'], device=device).unsqueeze(0)
            goals = torch.tensor(test_case['goal_poses'], device=device).unsqueeze(0)
            n_agents = torch.tensor([config['num_agents']], device=device)
            
            # Execute hierarchical trajectory
            results = controller.execute_hierarchical_trajectory(
                model, frame, starts, goals, n_agents, config['horizon'],
                render=True, store_frames=True
            )
            
            # Create outputs
            gif_name = f"hierarchical_case_{case_idx}.gif"
            controller.create_gif_from_frames(results, gif_name)
            
            all_results.append(results)
            
        except Exception as e:
            print(f"Error in test case {case_idx}: {e}")
            continue
    
    # Close dataset
    dataset_reader.close()
    
    # Print summary
    if all_results:
        success_rates = [r['agents_successful'].float().mean().item() for r in all_results]
        avg_success = np.mean(success_rates)
        print(f"\nOverall Results:")
        print(f"Average Success Rate: {avg_success:.2f}")
        print(f"Cases Processed: {len(all_results)}")
    
    return all_results, config

# Main execution
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Execute the system
    results, config = execute_hierarchical_moving_horizon_system(
        dataset_path="navigation_v2_Na_4_T_50_dataset.h5",
        model_path="boundary_constrained_madp_navigation_v2_na_4.pth",
        device=device,
        goal_tolerance=0.1,
        goal_pursuit_threshold=0.25,
        horizon_size=8,
        num_test_cases=10
    )
    
    if results is not None:
        print("\n🎉 Hierarchical Moving Horizon System executed successfully!")
        print("📊 Check the generated GIFs for detailed results.")
    else:
        print("❌ Execution failed. Please check the configuration and files.")
