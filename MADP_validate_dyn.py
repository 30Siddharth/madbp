"""
Modified MADP Validation System - No Dataset Loading, Random Noise Frames
Demonstrates three-step hierarchical control with synthetic visual inputs
"""

import torch
import numpy as np
import yaml
from PIL import Image
import time
import pyvirtualdisplay
from tqdm import tqdm

# Import required modules
from MADP_diffusion_v2_2 import EnhancedMultiAgentDiffusionModel
from vmas import make_env
from vmas.scenarios.navigation_v2_test import HeuristicPolicy

class ModifiedMADPValidator:
    """
    Modified MADP validation system without dataset dependency
    """
    
    def __init__(self, scenario_name="navigation_v2", device='cuda', max_agents=10):
        self.device = device
        self.scenario_name = scenario_name
        self.max_agents = max_agents
        self.horizon_steps = 10
        self.max_simulation_steps = 500
        
        # Initialize virtual display
        self.display = pyvirtualdisplay.Display(visible=False, size=(256, 256))
        self.display.start()
        
        # Load configuration and initialize components
        self._load_configuration()
        self._initialize_environment()
        self._load_model()
        
        print(f"Modified MADP Validator initialized")
        print(f"Using random noise frames instead of actual rendering")
    
    def _load_configuration(self):
        """Load training configuration"""
        try:
            with open("MADP_training_config.yaml", 'r') as file:
                config = yaml.safe_load(file)
                self.num_agents = config['param']['num_agents']
                self.horizon_length = config['param']['horizon']
                self.diffusion_steps = config['param']['diffuse_steps']
        except FileNotFoundError:
            print("Warning: Using default configuration")
            self.num_agents = 4
            self.horizon_length = 40
            self.diffusion_steps = 500
    
    def _initialize_environment(self):
        """Initialize VMAS environment"""
        self.env = make_env(
            scenario=f"{self.scenario_name}_test",
            num_envs=1,
            device=self.device,
            continuous_actions=True,
            max_steps=self.max_simulation_steps,
            n_agents=self.num_agents,
            collision_reward=-1.0,
            dist_shaping_factor=1.0,
            final_reward=0.01,
            agent_radius=0.05,
            shared_reward=True,
        )
        
        # Initialize storage lists
        self.env.frames = []
        self.env.actions = []
        self.env.observations = []
    
    def _load_model(self):
        """Load trained MADP model"""
        self.model = EnhancedMultiAgentDiffusionModel(
            max_agents=self.max_agents,
            horizon=8,
            state_dim=2,
            img_ch=3,
            hid=128,
            diffusion_steps=self.diffusion_steps,
            schedule_type='linear'
        ).to(self.device)
        
        model_path = f"boundary_constrained_madp_{self.scenario_name}_na_3.pth"
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found, using random weights")
        
        self.model.eval()
    
    def generate_random_noise_frame(self, shape=(128, 128, 3)):
        """
        Generate random noise frame matching the expected format
        
        Args:
            shape: Frame shape (H, W, C)
            
        Returns:
            Random noise frame as PIL Image and tensor
        """
        # Generate structured random noise
        noise = np.random.rand(*shape) * 255
        
        # Add some spatial structure to make it more realistic
        # Create circular patterns for "agents"
        for _ in range(np.random.randint(3, 8)):
            center_x = np.random.randint(50, shape[0]-50)
            center_y = np.random.randint(50, shape[1]-50)
            radius = np.random.randint(10, 25)
            
            y, x = np.ogrid[:shape[0], :shape[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            noise[mask] = np.random.rand(3) * 255
        
        noise_frame = noise.astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(noise_frame)
        
        # Convert to tensor format [C, H, W]
        frame_tensor = torch.tensor(noise_frame).permute(2, 0, 1).float()
        
        return pil_image, frame_tensor
    
    def invert_frame(self, frame):
        """
        Invert frame colors as in original validation code
        
        Args:
            frame: PIL Image
            
        Returns:
            Inverted frame as numpy array
        """
        frame = np.array(frame)
        # Rescale values between -1 and 1
        frame = frame / 127.5 - 1
        # Invert the colors
        frame = -frame
        # Convert back to range 0-255
        frame = ((frame + 1) * 127.5).astype(np.uint8)
        return frame
    
    def rendering_callback(self, env, td):
        """
        Modified rendering callback using random noise instead of actual rendering
        
        Args:
            env: VMAS environment
            td: Trajectory data
        """
        # Generate random noise frame instead of actual rendering
        noise_image, _ = self.generate_random_noise_frame()
        
        # Resize to match expected format
        noise_image = noise_image.resize((256, 256))
        env.frames.append(noise_image)
        
        # Store observations as before
        env.observations.append(td["agents"].cpu().numpy())
    
    def reset_environment_and_get_positions(self):
        """
        Reset environment and extract agent positions and goals
        
        Returns:
            Tuple of (start_positions, goal_positions, frame_tensor)
        """
        # Reset environment
        obs = self.env.reset()
        
        # Generate random noise frame for model input
        _, frame_tensor = self.generate_random_noise_frame((128, 128, 3))
        frame_batch = frame_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Extract start positions from environment
        start_positions = torch.zeros(1, 10, 2, device=self.device)
        goal_positions = torch.zeros(1, 10, 2, device=self.device)
        
        for i, agent in enumerate(self.env.agents):
            if i < self.num_agents:

                start_positions[0, i] = agent.state.pos[0]
                if hasattr(agent, 'goal'):
                    goal_positions[0, i] = agent.goal.state.pos[0]

                # Get current agent position
                # start_pos = agent.state.pos[0].cpu().numpy()
                # start_positions.append(start_pos)
                
                # # Get goal position (assuming navigation_v2 structure)
                # if hasattr(agent, 'goal'):
                #     goal_pos = agent.goal.state.pos[0].cpu().numpy()
                # else:
                #     # Generate random goal if not available
                #     goal_pos = np.random.uniform(-0.9, 0.9, 2)
                # goal_positions.append(goal_pos)
        
        # start_tensor = torch.tensor(start_positions, device=self.device).unsqueeze(0)
        # goal_tensor = torch.tensor(goal_positions, device=self.device).unsqueeze(0)
        
        return start_positions, goal_positions, frame_batch
    
    def generate_madp_trajectory(self, frame, starts, goals, n_agents):
        """
        Generate MADP trajectory using random noise frame
        
        Args:
            frame: Random noise frame tensor
            starts: Start positions
            goals: Goal positions
            n_agents: Number of agents
            
        Returns:
            Generated trajectory
        """

        assert starts.shape == (1, 10, 2), f"Expected [1,10,2], got {starts.shape}"
        assert goals.shape == (1, 10, 2), f"Expected [1,10,2], got {goals.shape}"
        assert frame.shape == (1, 3, 128, 128), f"Expected [1,3,128,128], got {frame.shape}"

        try:
            with torch.no_grad():
                trajectory = self.model.sample_with_constraints(
                    frame, starts, goals, 
                    torch.tensor([n_agents], device=self.device),
                    steps=50, 
                    max_step_size=0.1
                )
            return trajectory
        except Exception as e:
            print(f"Error generating MADP trajectory: {e}")
            # Return dummy trajectory
            return torch.randn(1, n_agents, self.horizon_length, 2, device=self.device)
    
    def execute_trajectory_with_three_step_control(self, trajectory, goal_positions):
        """
        Execute trajectory using three-step hierarchical control
        
        Args:
            trajectory: Generated trajectory from MADP
            goal_positions: Target goal positions
        """
        n_agents = trajectory.shape[1]
        import pdb
        pdb.set_trace()
        
        # Initialize CLF-QP controllers
        heuristic_controllers = []
        for i in range(n_agents):
            controller = HeuristicPolicy(
                continuous_action=True,
                clf_epsilon=0.2,
                clf_slack=10.0
            )
            heuristic_controllers.append(controller)
        
        # Control parameters
        goal_tolerance = 0.05
        trajectory_threshold = 0.5
        precision_threshold = 0.1
        Kp = 10.0
        Kv = 4.0
        max_force = 1.0
        
        # Execute trajectory
        for step in range(min(self.horizon_steps, trajectory.shape[2])):
            actions = []
            
            for i in range(n_agents):
                # Get current agent state
                current_pos = self.env.agents[i].state.pos[0]
                current_vel = self.env.agents[i].state.vel[0]
                goal_pos = goal_positions[0, i]
                
                # Calculate distance to goal
                distance = torch.norm(current_pos - goal_pos).item()
                
                if distance < goal_tolerance:
                    # Agent reached goal - stop
                    actions.append(torch.zeros(1, 2, device=self.device))
                elif distance >= trajectory_threshold:
                    # Step 1: MADP trajectory tracking
                    target_pos = trajectory[0, i, step]
                    pos_error = target_pos - current_pos
                    vel_error = -current_vel
                    
                    action = torch.clamp(
                        Kp * 1.2 * pos_error + Kv * vel_error,
                        -max_force, max_force
                    )
                    actions.append(action.unsqueeze(0))
                elif distance >= precision_threshold:
                    # Step 2: Goal pursuit
                    pos_error = goal_pos - current_pos
                    vel_error = -current_vel
                    
                    action = torch.clamp(
                        Kp * 0.8 * pos_error + Kv * 1.2 * vel_error,
                        -max_force, max_force
                    )
                    actions.append(action.unsqueeze(0))
                else:
                    # Step 3: Precision control with CLF-QP
                    try:
                        obs = self.env.scenario.observation(self.env.agents[i])[0]
                        obs_cpu = obs.detach().cpu().unsqueeze(0)
                        u_range = torch.tensor([max_force])
                        
                        action_cpu = heuristic_controllers[i].compute_action(obs_cpu, u_range)
                        action = action_cpu.squeeze(0).to(self.device)
                        actions.append(action.unsqueeze(0))
                    except:
                        # Fallback to gentle PD control
                        pos_error = goal_pos - current_pos
                        action = torch.clamp(
                            Kp * 0.5 * pos_error,
                            -max_force * 0.5, max_force * 0.5
                        )
                        actions.append(action.unsqueeze(0))
            
            # Step environment
            obs, rewards, dones, info = self.env.step(actions)
            
            # Store frame (random noise)
            self.rendering_callback(self.env, {"agents": obs})
    
    def run_validation_simulation(self, num_episodes=10):
        """
        Run complete validation simulation
        
        Args:
            num_episodes: Number of episodes to run
            
        Returns:
            Simulation results
        """
        print(f"Starting Modified MADP Validation Simulation")
        print(f"Episodes: {num_episodes}")
        print(f"Using random noise frames")
        
        results = {
            'episodes': [],
            'success_rates': [],
            'total_frames': 0
        }
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            # Reset environment and get positions
            starts, goals, frame = self.reset_environment_and_get_positions()
            
            print(f"Start positions: {starts[0].cpu().numpy()}")
            print(f"Goal positions: {goals[0].cpu().numpy()}")
            
            # Clear frame storage
            self.env.frames = []
            self.env.observations = []
            
            # Generate MADP trajectory
            trajectory = self.generate_madp_trajectory(
                frame, starts, goals, self.num_agents
            )
            
            # Execute with three-step control
            self.execute_trajectory_with_three_step_control(trajectory, goals)
            
            # Calculate success metrics
            final_distances = []
            for i in range(self.num_agents):
                current_pos = self.env.agents[i].state.pos[0]
                goal_pos = goals[0, i]
                distance = torch.norm(current_pos - goal_pos).item()
                final_distances.append(distance)
            
            success_count = sum(1 for d in final_distances if d < 0.05)
            success_rate = success_count / self.num_agents
            
            episode_result = {
                'episode': episode,
                'success_rate': success_rate,
                'final_distances': final_distances,
                'frames_generated': len(self.env.frames)
            }
            
            results['episodes'].append(episode_result)
            results['success_rates'].append(success_rate)
            results['total_frames'] += len(self.env.frames)
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Success rate: {success_rate:.2f}")
            print(f"  Final distances: {[f'{d:.3f}' for d in final_distances]}")
            print(f"  Frames generated: {len(self.env.frames)}")
        
        # Calculate overall statistics
        avg_success_rate = np.mean(results['success_rates'])
        std_success_rate = np.std(results['success_rates'])
        
        print(f"\n{'='*50}")
        print(f"VALIDATION SIMULATION COMPLETE")
        print(f"{'='*50}")
        print(f"Episodes completed: {num_episodes}")
        print(f"Average success rate: {avg_success_rate:.3f} ± {std_success_rate:.3f}")
        print(f"Total frames generated: {results['total_frames']}")
        print(f"Using random noise instead of actual rendering")
        
        return results
    
    def update_agent_population(self, current_n_active):
        """Handle dynamic agent spawning/removal"""
        # Remove agents that reached goals
        agents_to_remove = []
        for i in range(current_n_active):
            agent = self.env.agents[i]
            if hasattr(agent, 'goal'):
                distance = torch.norm(agent.state.pos[0] - agent.goal.state.pos[0]).item()
                if distance < 0.05:  # Goal reached
                    agents_to_remove.append(i)
        
        # Spawn new agents probabilistically
        new_n_active = current_n_active - len(agents_to_remove)
        if new_n_active < self.max_agents and np.random.rand() < 0.3:  # 30% spawn chance
            new_n_active += 1
        
        return new_n_active

    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'display'):
            self.display.stop()


def main():
    """Main execution function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize modified validator
    validator = ModifiedMADPValidator(
        scenario_name="navigation_v2",
        device=device,
        max_agents=10
    )
    
    # Run validation simulation
    results = validator.run_validation_simulation(num_episodes=10)
    
    # Cleanup
    validator.cleanup()
    
    print("\n✅ Modified MADP validation completed successfully!")
    return results


if __name__ == "__main__":
    results = main()
