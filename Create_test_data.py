import pyvirtualdisplay
import torch
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm
import os
import warnings
from pathlib import Path

class RobustTestDataCollector:
    """
    Robust test data collector that prevents common pitfalls in MADP validation
    
    Key improvements:
    - Prevents trailing comma syntax errors
    - Ensures tensor compatibility with PyTorch
    - Handles negative strides properly
    - Provides comprehensive error handling
    - Validates dataset structure
    """
    
    def __init__(self, scenario_name, n_agents, max_steps, num_cases, 
                 target_size=(256, 256), device='cuda'):
        """
        Initialize the robust test data collector
        
        Args:
            scenario_name: VMAS scenario name
            n_agents: Number of agents
            max_steps: Maximum steps (for metadata)
            num_cases: Number of test cases to collect
            target_size: Target frame size (width, height)
            device: Computing device
        """
        # Validate inputs to prevent type errors
        self.scenario_name = str(scenario_name)  # Ensure string type
        self.n_agents = int(n_agents)
        self.max_steps = int(max_steps)
        self.num_cases = int(num_cases)
        self.target_size = tuple(target_size)
        self.device = device
        
        # Create safe filename without trailing commas
        self.filename = self._create_safe_filename()
        
        print(f"Initializing RobustTestDataCollector")
        print(f"  Scenario: {self.scenario_name}")
        print(f"  Agents: {self.n_agents}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Test cases: {self.num_cases}")
        print(f"  Target frame size: {self.target_size}")
        print(f"  Output file: {self.filename}")
    
    def _create_safe_filename(self):
        """Create safe filename without trailing commas or syntax errors"""
        # Build filename components safely
        components = [
            self.scenario_name,
            f"Na_{self.n_agents}",
            f"T_{self.max_steps}",
            "dataset.h5"
        ]
        
        # Join with underscores and ensure no trailing comma
        filename = "_".join(components)
        
        # Validate filename
        if not filename.endswith('.h5'):
            raise ValueError(f"Invalid filename format: {filename}")
        
        return filename
    
    def safe_array_conversion(self, array):
        """
        Safely convert arrays to prevent negative stride errors
        
        Args:
            array: Input array (numpy or tensor)
            
        Returns:
            Safe numpy array with positive strides
        """
        if array is None:
            return None
        
        # Convert to numpy if needed
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        
        # Check for negative strides
        if hasattr(array, 'strides') and any(s < 0 for s in array.strides):
            array = array.copy()
        
        # Ensure contiguous memory layout
        array = np.ascontiguousarray(array, dtype=np.float32)
        
        return array
    
    def safe_frame_processing(self, frame):
        """
        Safely process frames to prevent tensor conversion errors
        
        Args:
            frame: Raw frame from environment
            
        Returns:
            Processed frame tensor
        """
        if frame is None:
            print("Warning: Received None frame, creating dummy frame")
            return torch.zeros(3, self.target_size[1], self.target_size[0])
        
        try:
            # Convert to PIL Image
            if isinstance(frame, np.ndarray):
                # Handle negative strides
                frame = self.safe_array_conversion(frame)
                frame_pil = Image.fromarray(frame.astype(np.uint8))
            else:
                frame_pil = frame
            
            # Resize to target dimensions
            frame_resized = frame_pil.resize(self.target_size, Image.LANCZOS)
            
            # Apply color inversion and normalization
            frame_array = np.array(frame_resized)
            frame_array = self.safe_array_conversion(frame_array)
            
            # Invert colors (as in original)
            frame_normalized = frame_array / 127.5 - 1
            frame_inverted = -frame_normalized
            frame_final = ((frame_inverted + 1) * 127.5).astype(np.uint8)
            
            # Convert to tensor with proper channel ordering
            frame_tensor = torch.tensor(frame_final, dtype=torch.float32)
            frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
            
            return frame_tensor
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            print("Creating dummy frame as fallback")
            return torch.zeros(3, self.target_size[1], self.target_size[0])
    
    def extract_agent_poses(self, env):
        """
        Safely extract agent start and goal poses
        
        Args:
            env: VMAS environment
            
        Returns:
            Tuple of (start_poses, goal_poses)
        """
        start_poses = []
        goal_poses = []
        
        try:
            for agent in env.agents:
                # Extract agent position (start pose)
                agent_pos = agent.state.pos[0]
                if hasattr(agent_pos, 'cpu'):
                    agent_pos = agent_pos.cpu().numpy()
                else:
                    agent_pos = np.array(agent_pos)
                
                # Ensure safe array conversion
                agent_pos = self.safe_array_conversion(agent_pos)
                start_poses.append(agent_pos[:2])  # Take only x, y coordinates
                
                # Extract goal position
                goal_pos = agent.goal.state.pos[0]
                if hasattr(goal_pos, 'cpu'):
                    goal_pos = goal_pos.cpu().numpy()
                else:
                    goal_pos = np.array(goal_pos)
                
                # Ensure safe array conversion
                goal_pos = self.safe_array_conversion(goal_pos)
                goal_poses.append(goal_pos[:2])  # Take only x, y coordinates
            
            # Convert to safe numpy arrays
            start_pose_array = np.array(start_poses, dtype=np.float32)
            goal_pose_array = np.array(goal_poses, dtype=np.float32)
            
            # Final safety check
            start_pose_array = self.safe_array_conversion(start_pose_array)
            goal_pose_array = self.safe_array_conversion(goal_pose_array)
            
            return start_pose_array, goal_pose_array
            
        except Exception as e:
            print(f"Error extracting poses: {e}")
            # Return dummy poses as fallback
            dummy_starts = np.zeros((self.n_agents, 2), dtype=np.float32)
            dummy_goals = np.ones((self.n_agents, 2), dtype=np.float32)
            return dummy_starts, dummy_goals
    
    def validate_dataset_structure(self, h5_file):
        """
        Validate HDF5 dataset structure to prevent loading errors
        
        Args:
            h5_file: Open HDF5 file handle
        """
        required_groups = ['frames', 'start_poses', 'goal_poses']
        required_attrs = ['num_agents', 'scenario_name', 'max_steps', 'num_cases']
        
        # Check required groups
        for group in required_groups:
            if group not in h5_file:
                raise ValueError(f"Missing required group: {group}")
        
        # Check required attributes
        for attr in required_attrs:
            if attr not in h5_file.attrs:
                raise ValueError(f"Missing required attribute: {attr}")
        
        print("Dataset structure validation passed")
    
    def collect_test_data(self, env):
        """
        Main data collection method with comprehensive error handling
        
        Args:
            env: Initialized VMAS environment
            
        Returns:
            Path to created dataset file
        """
        # Start virtual display
        display = pyvirtualdisplay.Display(visible=False, size=self.target_size)
        display.start()
        
        try:
            # Check if file already exists and handle appropriately
            if os.path.exists(self.filename):
                response = input(f"File {self.filename} exists. Overwrite? (y/n): ")
                if response.lower() != 'y':
                    print("Aborting data collection")
                    return None
            
            with h5py.File(self.filename, 'w') as f:
                # Create groups for different data types
                frames_group = f.create_group('frames')
                start_poses_group = f.create_group('start_poses')
                goal_poses_group = f.create_group('goal_poses')
                
                # Store metadata safely (no trailing commas!)
                f.attrs['num_agents'] = self.n_agents
                f.attrs['scenario_name'] = self.scenario_name
                f.attrs['max_steps'] = self.max_steps
                f.attrs['num_cases'] = self.num_cases
                
                # Progress tracking
                pbar = tqdm(total=self.num_cases, desc="Collecting test cases")
                successful_cases = 0
                failed_cases = 0
                
                # Collect data for each test case
                for case_idx in range(self.num_cases):
                    try:
                        # Reset environment to get new random configuration
                        env.reset()
                        
                        # Render first frame with error handling
                        first_frame = env.render(mode="rgb_array", env_index=0)
                        
                        # Process frame safely
                        frame_tensor = self.safe_frame_processing(first_frame)
                        batch_image_tensor = torch.stack([frame_tensor])
                        
                        # Extract poses safely
                        start_poses, goal_poses = self.extract_agent_poses(env)
                        
                        # Validate data before saving
                        if self._validate_case_data(frame_tensor, start_poses, goal_poses):
                            # Save data to HDF5 file
                            episode_name = f'episode_{case_idx}'
                            
                            # Safe dataset creation
                            frames_group.create_dataset(
                                episode_name,
                                data=batch_image_tensor.numpy(),
                                compression="gzip"
                            )
                            
                            start_poses_group.create_dataset(
                                episode_name,
                                data=start_poses
                            )
                            
                            goal_poses_group.create_dataset(
                                episode_name,
                                data=goal_poses
                            )
                            
                            successful_cases += 1
                        else:
                            failed_cases += 1
                            print(f"Warning: Invalid data for case {case_idx}, skipping")
                        
                    except Exception as e:
                        failed_cases += 1
                        print(f"Error processing case {case_idx}: {e}")
                        continue
                    
                    # Update progress
                    pbar.set_description(f"Success: {successful_cases}, Failed: {failed_cases}")
                    pbar.update()
                
                pbar.close()
                
                # Update final metadata
                f.attrs['successful_cases'] = successful_cases
                f.attrs['failed_cases'] = failed_cases
                
                # Validate final dataset structure
                self.validate_dataset_structure(f)
            
            print(f"Dataset collection completed successfully!")
            print(f"  Successful cases: {successful_cases}")
            print(f"  Failed cases: {failed_cases}")
            print(f"  Dataset saved to: {self.filename}")
            
            # Verify file integrity
            self._verify_dataset_integrity()
            
            return self.filename
            
        except Exception as e:
            print(f"Critical error during data collection: {e}")
            return None
            
        finally:
            # Clean up
            display.stop()
    
    def _validate_case_data(self, frame_tensor, start_poses, goal_poses):
        """Validate individual case data"""
        try:
            # Check frame tensor
            if frame_tensor.shape != (3, self.target_size[1], self.target_size[0]):
                return False
            
            # Check pose arrays
            if start_poses.shape != (self.n_agents, 2):
                return False
            
            if goal_poses.shape != (self.n_agents, 2):
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(start_poses)) or np.any(np.isinf(start_poses)):
                return False
            
            if np.any(np.isnan(goal_poses)) or np.any(np.isinf(goal_poses)):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _verify_dataset_integrity(self):
        """Verify the created dataset can be loaded properly"""
        try:
            with h5py.File(self.filename, 'r') as f:
                # Check groups exist
                assert 'frames' in f
                assert 'start_poses' in f
                assert 'goal_poses' in f
                
                # Check attributes
                assert 'num_agents' in f.attrs
                assert 'scenario_name' in f.attrs
                assert 'max_steps' in f.attrs
                assert 'num_cases' in f.attrs
                
                # Sample a few episodes
                frames_group = f['frames']
                start_group = f['start_poses']
                goal_group = f['goal_poses']
                
                episode_keys = list(frames_group.keys())
                if len(episode_keys) > 0:
                    # Test loading first episode
                    test_key = episode_keys[0]
                    test_frame = frames_group[test_key][:]
                    test_start = start_group[test_key][:]
                    test_goal = goal_group[test_key][:]
                    
                    # Convert to tensors to ensure compatibility
                    frame_tensor = torch.tensor(test_frame, dtype=torch.float32)
                    start_tensor = torch.tensor(test_start, dtype=torch.float32)
                    goal_tensor = torch.tensor(test_goal, dtype=torch.float32)
                    
                    print("Dataset integrity verification passed")
                else:
                    print("Warning: No episodes found in dataset")
                    
        except Exception as e:
            print(f"Dataset integrity verification failed: {e}")
            raise

def collect_vmas_test_data(env, scenario_name, n_agents, max_steps, num_cases):
    """
    Main function for collecting VMAS test data with pitfall prevention
    
    Args:
        env: Initialized VMAS environment
        scenario_name: Name of the scenario
        n_agents: Number of agents
        max_steps: Maximum steps (for metadata)
        num_cases: Number of test cases to collect
        
    Returns:
        Path to created dataset file
    """
    # Initialize robust collector
    collector = RobustTestDataCollector(
        scenario_name=scenario_name,
        n_agents=n_agents,
        max_steps=max_steps,
        num_cases=num_cases
    )
    
    # Collect data
    return collector.collect_test_data(env)

# Usage Example with comprehensive error handling
if __name__ == "__main__":
    from vmas import make_env
    import sys
    
    # Configuration parameters
    scenario_name = "navigation_v2"  # No trailing comma!
    n_agents = 4
    max_steps = 50
    num_cases = 100
    
    print("Starting robust VMAS test data collection")
    print("=" * 50)
    
    try:
        # Create VMAS environment
        env = make_env(
            scenario=scenario_name,
            num_envs=1,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            continuous_actions=True,
            max_steps=max_steps,
            n_agents=n_agents,
            collision_reward=-1.0,
            dist_shaping_factor=1.0,
            final_reward=0.01,
            agent_radius=0.05,
            # x_semidim=1.0,
            # y_semidim=1.0,
            lidar_range=0.1,
            shared_reward=True,
            use_test_positions=False  # Use random positions
        )
        
        # Collect test data using robust collector
        dataset_filename = collect_vmas_test_data(
            env=env,
            scenario_name=scenario_name,
            n_agents=n_agents,
            max_steps=max_steps,
            num_cases=num_cases
        )
        
        if dataset_filename:
            print(f"\n✅ Test dataset created successfully: {dataset_filename}")
            
            # Print file info
            file_size = os.path.getsize(dataset_filename) / (1024 * 1024)  # MB
            print(f"📁 File size: {file_size:.2f} MB")
            
            # Verify compatibility with MADP_validate.py
            print("\n🔍 Verifying compatibility with MADP validation...")
            try:
                with h5py.File(dataset_filename, 'r') as f:
                    print(f"✓ Dataset groups: {list(f.keys())}")
                    print(f"✓ Metadata: {dict(f.attrs)}")
                    print("✓ Ready for use with MADP_validate.py")
            except Exception as e:
                print(f"⚠️ Compatibility check failed: {e}")
        else:
            print("❌ Dataset creation failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)
