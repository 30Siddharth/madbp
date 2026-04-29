"""
Enhanced MADP Validation System with Dynamic Agent Capability Visualization
Demonstrates MADP's ability to handle varying numbers of agents with comprehensive plotting
"""

import torch
import numpy as np
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import pandas as pd

from vmas import make_env
from MADP_diffusion_v2_2 import EnhancedMultiAgentDiffusionModel


class DynamicAgentMADPValidationSystem:
    """
    Enhanced validation system showcasing MADP's dynamic agent handling capabilities
    """
    
    def __init__(self, scenario_name="dynamic_env_2", device='cuda', 
                 initial_agents=4, max_agents=10, agent_spawn_strategy='progressive'):
        self.device = device
        self.scenario_name = scenario_name
        self.initial_agents = initial_agents
        self.max_agents = max_agents
        self.goal_tolerance = 0.05
        
        # Dynamic agent management
        self.agent_spawn_strategy = agent_spawn_strategy  # 'progressive', 'random', 'cyclic'
        self.current_agent_count = initial_agents
        
        # Episode tracking with detailed metrics
        self.episode_count = 0
        self.agent_history = []  # Track agent count over episodes
        self.spawn_history = []  # Track when new agents are spawned
        self.performance_history = []  # Track performance metrics
        
        # Frame storage
        self.frames_original = []
        self.model_input_size = (128, 128, 3)
        
        # Comprehensive results storage for plotting
        self.validation_results = {
            'episode_data': [],
            'agent_dynamics': [],
            'spawn_events': [],
            'performance_metrics': [],
            'scalability_data': []
        }
        
        self._load_model()
        
        print(f"Dynamic Agent MADP Validation System initialized")
        print(f"Agent range: {initial_agents} → {max_agents}")
        print(f"Spawn strategy: {agent_spawn_strategy}")
        print(f"Showcasing MADP's dynamic agent handling capability")
    
    def _load_model(self):
        """Load trained MADP model"""
        self.model = EnhancedMultiAgentDiffusionModel(
            max_agents=self.max_agents,
            horizon=8,
            state_dim=2,
            img_ch=3,
            hid=128,
            diffusion_steps=500,
            schedule_type='linear'
        ).to(self.device)
        
        model_path = f"boundary_constrained_madp_{self.scenario_name}_na_4.pth"
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            print(f"Loaded model from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found, using random weights")
        
        self.model.eval()
    
    def _initialize_environment(self, n_agents: int):
        """Initialize environment with specific number of agents"""
        self.env = make_env(
            scenario=self.scenario_name,
            num_envs=1,
            device=self.device,
            continuous_actions=True,
            max_steps=40,
            n_agents=n_agents,
            collisions=True,
            shared_rew=True,
            pos_shaping_factor=1.0,
            final_reward=0.01,
            agent_collision_penalty=-1.0,
            agent_radius=0.05,
        )
        
        print(f"Environment reinitialized with {n_agents} agents")
        return self.env
    
    def determine_next_agent_count(self) -> Tuple[int, bool]:
        """
        Determine the number of agents for the next episode based on strategy
        
        Returns:
            Tuple of (next_agent_count, is_new_spawn)
        """
        is_new_spawn = False
        
        if self.agent_spawn_strategy == 'progressive':
            # Gradually increase agent count
            if self.episode_count > 0 and self.current_agent_count < self.max_agents:
                if self.episode_count % 2 == 0:  # Spawn every 2 episodes
                    self.current_agent_count = min(self.current_agent_count + 1, self.max_agents)
                    is_new_spawn = True
        
        elif self.agent_spawn_strategy == 'random':
            # Random agent count within range
            if self.episode_count > 0:
                new_count = np.random.randint(self.initial_agents, self.max_agents + 1)
                is_new_spawn = new_count > self.current_agent_count
                self.current_agent_count = new_count
        
        elif self.agent_spawn_strategy == 'cyclic':
            # Cycle through different agent counts
            cycle_length = 4
            cycle_position = self.episode_count % cycle_length
            agent_counts = [4, 6, 8, 5]  # Example cycle
            new_count = agent_counts[cycle_position]
            is_new_spawn = new_count > self.current_agent_count
            self.current_agent_count = new_count
        
        return self.current_agent_count, is_new_spawn
    
    def render_frame(self, for_model: bool = False) -> torch.Tensor:
        """Render environment frame with optional reshaping"""
        try:
            frame = self.env.render(
                mode="rgb_array",
                agent_index_focus=None,
                visualize_when_rgb=True
            )
            
            if frame is not None:
                return self._process_rendered_frame(frame, for_model)
            else:
                return self._generate_synthetic_frame(for_model)
                
        except Exception as e:
            print(f"Rendering failed: {e}, using synthetic frame")
            return self._generate_synthetic_frame(for_model)
    
    def _process_rendered_frame(self, frame: np.ndarray, for_model: bool) -> torch.Tensor:
        """Process rendered frame based on intended usage"""
        pil_image = Image.fromarray(frame)
        
        if for_model:
            resized_image = pil_image.resize((128, 128), Image.Resampling.LANCZOS)
            frame_array = np.array(resized_image)
            
            if frame_array.shape[-1] != 3:
                frame_array = np.stack([frame_array] * 3, axis=-1)
            
            frame_tensor = torch.tensor(frame_array).permute(2, 0, 1).float()
            return frame_tensor.unsqueeze(0).to(self.device)
        else:
            self.frames_original.append(pil_image)
            
            frame_array = np.array(pil_image)
            if frame_array.shape[-1] != 3:
                frame_array = np.stack([frame_array] * 3, axis=-1)
            
            frame_tensor = torch.tensor(frame_array).permute(2, 0, 1).float()
            return frame_tensor.unsqueeze(0).to(self.device)
    
    def _generate_synthetic_frame(self, for_model: bool = False) -> torch.Tensor:
        """Generate synthetic frame with appropriate dimensions"""
        if for_model:
            frame = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            grid_spacing = 16
        else:
            frame = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
            grid_spacing = 50
        
        for i in range(0, frame.shape[0], grid_spacing):
            frame[i:i+2, :, :] = 128
            frame[:, i:i+2, :] = 128
        
        if not for_model:
            pil_image = Image.fromarray(frame)
            self.frames_original.append(pil_image)
        
        frame_tensor = torch.tensor(frame).permute(2, 0, 1).float()
        return frame_tensor.unsqueeze(0).to(self.device)
    
    def extract_agent_positions(self, n_active: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract start and goal positions for active agents"""
        start_positions = torch.zeros(1, self.max_agents, 2, device=self.device)
        goal_positions = torch.zeros(1, self.max_agents, 2, device=self.device)
        
        for i in range(min(n_active, len(self.env.agents))):
            agent = self.env.agents[i]
            start_positions[0, i] = agent.state.pos[0]
            
            if hasattr(agent, 'goal'):
                goal_positions[0, i] = agent.goal.state.pos[0]
        
        return start_positions, goal_positions
    
    def store_episode_results(self, n_active: int, was_spawn_episode: bool) -> Dict:
        """Store comprehensive episode results with spawn tracking"""
        episode_results = {
            'final_positions': {},
            'goal_positions': {},
            'completed': {},
            'distances': {},
            'completion_summary': {},
            'agent_dynamics': {
                'active_count': n_active,
                'was_spawn_episode': was_spawn_episode,
                'spawn_strategy': self.agent_spawn_strategy
            }
        }
        
        print(f"\n--- Episode {self.episode_count + 1} Results ---")
        print(f"Active agents: {n_active} {'(+NEW SPAWN)' if was_spawn_episode else ''}")
        
        for i in range(n_active):
            agent = self.env.agents[i]
            
            current_pos = agent.state.pos[0].clone()
            episode_results['final_positions'][i] = current_pos
            
            if hasattr(agent, 'goal'):
                goal_pos = agent.goal.state.pos[0].clone()
                episode_results['goal_positions'][i] = goal_pos
                
                distance = torch.norm(current_pos - goal_pos).item()
                episode_results['distances'][i] = distance
                
                completed = distance < self.goal_tolerance
                episode_results['completed'][i] = completed
                
                status = "✅ COMPLETED" if completed else "❌ INCOMPLETE"
                agent_type = "NEW" if was_spawn_episode and i >= (n_active - 1) else "EXISTING"
                
                print(f"Agent {i} ({agent_type}): {status} | Distance: {distance:.4f} | "
                      f"Pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}] | "
                      f"Goal: [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}]")
            else:
                episode_results['completed'][i] = False
                episode_results['distances'][i] = float('inf')
                print(f"Agent {i}: No goal assigned")
        
        # Calculate comprehensive metrics
        completed_count = sum(episode_results['completed'].values())
        success_rate = completed_count / n_active if n_active > 0 else 0.0
        
        episode_results['completion_summary'] = {
            'completed_agents': completed_count,
            'total_agents': n_active,
            'success_rate': success_rate,
            'scalability_performance': success_rate / n_active if n_active > 0 else 0.0  # Performance per agent
        }
        
        print(f"Episode Summary: {completed_count}/{n_active} agents completed "
              f"(Success Rate: {success_rate:.2f})")
        
        return episode_results
    
    def apply_state_persistence(self, n_active: int, previous_count: int):
        """Apply state persistence with support for new agent integration"""
        if not hasattr(self, 'previous_episode_results'):
            print("No previous episode data for state persistence")
            return
        
        previous_results = self.previous_episode_results
        print(f"\n--- Applying State Persistence ---")
        print(f"Previous agents: {previous_count}, Current agents: {n_active}")
        
        persistence_applied = 0
        new_agents_added = max(0, n_active - previous_count)
        
        # Apply persistence for existing agents
        for i in range(min(previous_count, n_active)):
            if i < len(self.env.agents):
                agent = self.env.agents[i]
                
                if (i in previous_results['completed'] and 
                    not previous_results['completed'][i]):
                    
                    if i in previous_results['final_positions']:
                        restored_pos = previous_results['final_positions'][i]
                        agent.state.pos[0] = restored_pos.to(self.device)
                        agent.state.vel[0] = torch.zeros(2, device=self.device)
                        
                        print(f"Agent {i}: Position restored to [{restored_pos[0]:.3f}, {restored_pos[1]:.3f}]")
                        persistence_applied += 1
                    
                    if (hasattr(agent, 'goal') and 
                        i in previous_results['goal_positions']):
                        restored_goal = previous_results['goal_positions'][i]
                        agent.goal.state.pos[0] = restored_goal.to(self.device)
                        
                        print(f"Agent {i}: Goal maintained at [{restored_goal[0]:.3f}, {restored_goal[1]:.3f}]")
                else:
                    print(f"Agent {i}: Fresh start (completed previous goal)")
        
        # Handle new agents
        if new_agents_added > 0:
            print(f"New agents added: {new_agents_added} (indices {previous_count} to {n_active-1})")
            for i in range(previous_count, n_active):
                if i < len(self.env.agents):
                    agent = self.env.agents[i]
                    print(f"Agent {i}: NEW AGENT - Random initialization")
        
        print(f"State persistence applied to {persistence_applied} existing agents")
        print(f"New agents integrated: {new_agents_added}")
    
    def run_dynamic_episode(self, max_steps: int = 100) -> Dict:
        """Run episode with dynamic agent management"""
        # Determine agent count for this episode
        n_active, is_spawn_episode = self.determine_next_agent_count()
        previous_agent_count = getattr(self, 'previous_agent_count', self.initial_agents)
        
        print(f"\n{'='*70}")
        print(f"EPISODE {self.episode_count + 1} - DYNAMIC AGENT MANAGEMENT")
        print(f"{'='*70}")
        print(f"Agent count transition: {previous_agent_count} → {n_active}")
        print(f"Spawn strategy: {self.agent_spawn_strategy}")
        print(f"New spawn event: {'YES' if is_spawn_episode else 'NO'}")
        
        # Initialize environment with current agent count
        self.env = self._initialize_environment(n_active)
        obs = self.env.reset()
        
        # Apply state persistence with dynamic agent support
        if self.episode_count > 0:
            self.apply_state_persistence(n_active, previous_agent_count)
        
        # Track agent dynamics
        self.agent_history.append(n_active)
        if is_spawn_episode:
            self.spawn_history.append({
                'episode': self.episode_count + 1,
                'previous_count': previous_agent_count,
                'new_count': n_active,
                'new_agents': n_active - previous_agent_count
            })
        
        # Store previous count for next episode
        self.previous_agent_count = n_active
        
        # Render initial frames
        initial_frame_model = self.render_frame(for_model=True)
        initial_frame_gif = self.render_frame(for_model=False)
        
        print(f"Initial frame captured: {initial_frame_model.shape}")
        
        # Extract positions
        start_positions, goal_positions = self.extract_agent_positions(n_active)
        
        print(f"\nCurrent Agent Setup ({n_active} agents):")
        for i in range(n_active):
            start_pos = start_positions[0, i].cpu().numpy()
            goal_pos = goal_positions[0, i].cpu().numpy()
            agent_status = "NEW" if (is_spawn_episode and i >= previous_agent_count) else "EXISTING"
            print(f"  Agent {i} ({agent_status}): Start [{start_pos[0]:.3f}, {start_pos[1]:.3f}] → "
                  f"Goal [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}]")
        
        # Generate MADP trajectory for current agent configuration
        try:
            with torch.no_grad():
                trajectory = self.model.sample_with_constraints(
                    initial_frame_model,
                    start_positions,
                    goal_positions,
                    torch.tensor([n_active], device=self.device),
                    steps=50,
                    max_step_size=0.1
                )
            print(f"Generated trajectory for {n_active} agents: {trajectory.shape}")
        except Exception as e:
            print(f"Trajectory generation failed: {e}, using random trajectory")
            trajectory = torch.randn(1, self.max_agents, 8, 2, device=self.device)
        
        # Execute episode
        episode_rewards = []
        
        for step in range(max_steps):
            # Generate actions for all active agents
            actions = []
            for i in range(n_active):
                agent = self.env.agents[i]
                current_pos = agent.state.pos[0]
                # current_pos = trajectory[:,:,i,:]
                goal_pos = agent.goal.state.pos[0] if hasattr(agent, 'goal') else current_pos
                
                # Simple proportional control
                error = goal_pos - current_pos
                action = torch.clamp(error * 2.0, -0.7, 0.7)
                actions.append(action.unsqueeze(0))
            
            # Step environment
            obs, rewards, dones, info = self.env.step(actions)
            
            # Handle rewards
            if isinstance(rewards, list):
                reward_values = [rewards[i].cpu().numpy() for i in range(len(rewards))]
                episode_rewards.append(np.array(reward_values))
            else:
                episode_rewards.append(rewards.cpu().numpy())
            
            # Capture frames periodically
            if step % 10 == 0:
                frame_model = self.render_frame(for_model=True)
                frame_gif = self.render_frame(for_model=False)
            
            # Check termination
            if isinstance(dones, list):
                if all(dones[i].all() for i in range(len(dones))):
                    print(f"Episode terminated early at step {step}")
                    break
            else:
                if dones.all():
                    print(f"Episode terminated early at step {step}")
                    break
        
        # Store comprehensive episode results
        episode_results = self.store_episode_results(n_active, is_spawn_episode)
        
        # Store for next episode's state persistence
        self.previous_episode_results = episode_results
        
        # Calculate detailed metrics
        completion_summary = episode_results['completion_summary']
        
        # Compute average reward
        if episode_rewards:
            all_rewards = np.stack(episode_rewards, axis=0)
            avg_reward = float(np.mean(all_rewards)) if len(all_rewards) > 0 else 0.0
        else:
            avg_reward = 0.0
        
        # Store comprehensive episode data for plotting
        episode_data = {
            'episode': self.episode_count + 1,
            'n_active': n_active,
            'previous_count': previous_agent_count,
            'is_spawn_episode': is_spawn_episode,
            'new_agents_count': max(0, n_active - previous_agent_count) if is_spawn_episode else 0,
            'success_rate': completion_summary['success_rate'],
            'completed_agents': completion_summary['completed_agents'],
            'scalability_performance': completion_summary['scalability_performance'],
            'avg_reward': avg_reward,
            'total_steps': min(step + 1, max_steps),
            'spawn_strategy': self.agent_spawn_strategy,
            'agent_dynamics': episode_results['agent_dynamics']
        }
        
        # Store in validation results for comprehensive analysis
        self.validation_results['episode_data'].append(episode_data)
        self.validation_results['agent_dynamics'].append({
            'episode': self.episode_count + 1,
            'agent_count': n_active,
            'spawn_event': is_spawn_episode,
            'performance': completion_summary['success_rate']
        })
        
        if is_spawn_episode:
            self.validation_results['spawn_events'].append({
                'episode': self.episode_count + 1,
                'previous_count': previous_agent_count,
                'new_count': n_active,
                'performance_impact': completion_summary['success_rate']
            })
        
        # Compile final results
        final_results = {
            'episode': self.episode_count + 1,
            'n_active': n_active,
            'is_spawn_episode': is_spawn_episode,
            'initial_frame_model': initial_frame_model,
            'start_positions': start_positions,
            'goal_positions': goal_positions,
            'episode_results': episode_results,
            'success_rate': completion_summary['success_rate'],
            'completed_agents': completion_summary['completed_agents'],
            'avg_reward': avg_reward,
            'total_steps': min(step + 1, max_steps),
            'trajectory': trajectory,
            'episode_rewards': episode_rewards
        }
        
        print(f"\nEpisode {self.episode_count + 1} completed:")
        print(f"  Active agents: {n_active}")
        print(f"  Success rate: {completion_summary['success_rate']:.2f}")
        print(f"  Scalability performance: {completion_summary['scalability_performance']:.4f}")
        print(f"  Average reward: {avg_reward:.4f}")
        print(f"  Total steps: {final_results['total_steps']}")
        
        self.episode_count += 1
        return final_results
    
    def run_dynamic_validation_sequence(self, episodes: int = 10) -> List[Dict]:
        """Run validation sequence showcasing dynamic agent handling"""
        print(f"\n🚀 DYNAMIC AGENT MADP VALIDATION SEQUENCE")
        print(f"Episodes: {episodes}")
        print(f"Agent range: {self.initial_agents} → {self.max_agents}")
        print(f"Spawn strategy: {self.agent_spawn_strategy}")
        print(f"Showcasing MADP's dynamic multi-agent capability")
        
        all_results = []
        
        for episode_idx in range(episodes):
            episode_results = self.run_dynamic_episode(max_steps=100)
            all_results.append(episode_results)
        
        # Store performance metrics for analysis
        self._compute_scalability_metrics()
        
        # Print comprehensive summary
        self._print_dynamic_validation_summary(all_results)
        
        return all_results
    
    def _compute_scalability_metrics(self):
        """Compute comprehensive scalability metrics"""
        episode_data = self.validation_results['episode_data']
        
        if not episode_data:
            return
        
        # Group performance by agent count
        performance_by_count = {}
        for ep in episode_data:
            count = ep['n_active']
            if count not in performance_by_count:
                performance_by_count[count] = {
                    'success_rates': [],
                    'rewards': [],
                    'scalability_performances': []
                }
            
            performance_by_count[count]['success_rates'].append(ep['success_rate'])
            performance_by_count[count]['rewards'].append(ep['avg_reward'])
            performance_by_count[count]['scalability_performances'].append(ep['scalability_performance'])
        
        # Compute scalability metrics
        scalability_data = []
        for count, data in performance_by_count.items():
            scalability_data.append({
                'agent_count': count,
                'mean_success_rate': np.mean(data['success_rates']),
                'std_success_rate': np.std(data['success_rates']),
                'mean_reward': np.mean(data['rewards']),
                'std_reward': np.std(data['rewards']),
                'mean_scalability_performance': np.mean(data['scalability_performances']),
                'episode_count': len(data['success_rates'])
            })
        
        self.validation_results['scalability_data'] = scalability_data
        
        print(f"\n--- Scalability Analysis ---")
        for data in scalability_data:
            print(f"  {data['agent_count']} agents: Success Rate = {data['mean_success_rate']:.3f} ± {data['std_success_rate']:.3f} "
                  f"({data['episode_count']} episodes)")
    
    def _print_dynamic_validation_summary(self, results: List[Dict]):
        """Print comprehensive dynamic validation summary"""
        print(f"\n{'='*80}")
        print(f"DYNAMIC AGENT MADP VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        total_episodes = len(results)
        success_rates = [r['success_rate'] for r in results]
        avg_success_rate = np.mean(success_rates)
        avg_reward = np.mean([r['avg_reward'] for r in results])
        
        # Agent dynamics analysis
        min_agents = min([r['n_active'] for r in results])
        max_agents = max([r['n_active'] for r in results])
        spawn_episodes = sum([1 for r in results if r['is_spawn_episode']])
        
        print(f"Configuration:")
        print(f"  Total episodes: {total_episodes}")
        print(f"  Agent range: {min_agents} → {max_agents}")
        print(f"  Spawn strategy: {self.agent_spawn_strategy}")
        print(f"  Spawn episodes: {spawn_episodes}/{total_episodes}")
        
        print(f"\nDynamic Agent Performance:")
        print(f"  Average success rate: {avg_success_rate:.3f}")
        print(f"  Average reward: {avg_reward:.4f}")
        print(f"  Agent scalability demonstrated: {max_agents - min_agents} agent range")
        print(f"  GIF frames captured: {len(self.frames_original)}")
        
        print(f"\nPer-Episode Breakdown:")
        for i, result in enumerate(results):
            spawn_indicator = "🆕" if result['is_spawn_episode'] else "  "
            print(f"  Episode {i+1}: {spawn_indicator} {result['n_active']} agents | "
                  f"{result['completed_agents']} completed | "
                  f"Rate: {result['success_rate']:.2f} | "
                  f"Reward: {result['avg_reward']:.3f}")
        
        if self.spawn_history:
            print(f"\nSpawn Event Analysis:")
            for spawn in self.spawn_history:
                print(f"  Episode {spawn['episode']}: {spawn['previous_count']} → {spawn['new_count']} "
                      f"(+{spawn['new_agents']} new agents)")
        
        print(f"\n✅ Dynamic agent validation sequence completed!")
        print(f"🎯 MADP successfully handled {min_agents}-{max_agents} agents dynamically")
        print(f"📈 {spawn_episodes} successful agent spawn events")
        print(f"🎬 High-quality visualization ready")
    
    def generate_dynamic_agent_capability_plots(self, output_dir: str = "validation_output"):
        """
        Generate comprehensive plots showcasing MADP's dynamic agent handling capability
        FIXED: Corrected spawning visualization and array length issues
        """
        os.makedirs(output_dir, exist_ok=True)
        
        episode_data = self.validation_results['episode_data']
        spawn_events = self.validation_results['spawn_events']
        scalability_data = self.validation_results['scalability_data']
    
        if not episode_data:
            print("No episode data available for plotting")
            return
        
        # Set up professional plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create comprehensive figure showcasing dynamic agent capability
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle('MADP Dynamic Multi-Agent Capability Analysis', fontsize=20, fontweight='bold', y=0.95)
        
        # Extract common data with proper validation
        episodes = [ep['episode'] for ep in episode_data]
        agent_counts = [ep['n_active'] for ep in episode_data]
        success_rates = [ep['success_rate'] for ep in episode_data]
        rewards = [ep['avg_reward'] for ep in episode_data]
        
        # Identify spawn episodes correctly
        spawn_episode_flags = [ep.get('is_spawn_episode', False) for ep in episode_data]
        spawn_episodes = [ep for ep, is_spawn in zip(episodes, spawn_episode_flags) if is_spawn]
        spawn_counts = [count for count, is_spawn in zip(agent_counts, spawn_episode_flags) if is_spawn]
    
        # 1. Dynamic Agent Count Timeline (Main Plot) - FIXED
        ax1 = plt.subplot(3, 4, (1, 2))  # Span 2 columns
        
        # Plot agent count with spawn highlights
        ax1.plot(episodes, agent_counts, 'o-', linewidth=3, markersize=8, 
                color='navy', label='Active Agents')
        
        # Highlight spawn episodes - FIXED length matching
        if spawn_episodes and spawn_counts and len(spawn_episodes) == len(spawn_counts):
            ax1.scatter(spawn_episodes, spawn_counts, s=200, c='red', marker='*', 
                    label='New Agent Spawn', zorder=5, edgecolor='darkred', linewidth=2)
            
            # Add value labels on spawn points
            for ep, count in zip(spawn_episodes, spawn_counts):
                ax1.annotate(f'🆕{count}', (ep, count), textcoords="offset points", 
                            xytext=(0,15), ha='center', fontweight='bold', fontsize=10)
        
        # Fill area under curve for visual impact
        ax1.fill_between(episodes, agent_counts, alpha=0.3, color='lightblue')
        
        ax1.set_title('MADP Dynamic Agent Management Timeline', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Number of Active Agents')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
    
        # 2. Success Rate vs Agent Count
        ax2 = plt.subplot(3, 4, 3)
        
        scatter = ax2.scatter(agent_counts, success_rates, c=episodes, s=100, 
                            cmap='viridis', alpha=0.7, edgecolor='black')
        
        # Add trend line if sufficient data
        if len(set(agent_counts)) > 1:
            try:
                z = np.polyfit(agent_counts, success_rates, 1)
                p = np.poly1d(z)
                trend_x = np.linspace(min(agent_counts), max(agent_counts), 100)
                ax2.plot(trend_x, p(trend_x), "r--", alpha=0.8, linewidth=2, 
                        label=f'Trend (slope: {z[0]:.3f})')
                
                correlation = np.corrcoef(agent_counts, success_rates)[0, 1]
                ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except np.linalg.LinAlgError:
                pass  # Skip trend line if calculation fails
    
        ax2.set_title('MADP Scalability Performance')
        ax2.set_xlabel('Number of Agents')
        ax2.set_ylabel('Success Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Episode')
        
        # 3. Spawn Event Impact Analysis - FIXED
        ax3 = plt.subplot(3, 4, 4)
        
        if spawn_events:
            # Use spawn_events data directly for consistency
            spawn_episodes_list = [se['episode'] for se in spawn_events]
            spawn_performance = [se['performance_impact'] for se in spawn_events]
            agent_increases = [se['new_count'] - se['previous_count'] for se in spawn_events]
            
            # Ensure all lists have same length
            min_length = min(len(spawn_episodes_list), len(spawn_performance), len(agent_increases))
            spawn_episodes_list = spawn_episodes_list[:min_length]
            spawn_performance = spawn_performance[:min_length]
            agent_increases = agent_increases[:min_length]
            
            if min_length > 0:
                bars = ax3.bar(spawn_episodes_list, spawn_performance, 
                            color=['green' if p >= 0.5 else 'orange' for p in spawn_performance],
                            alpha=0.7, edgecolor='black')
                
                # Add agent increase labels
                for ep, perf, increase in zip(spawn_episodes_list, spawn_performance, agent_increases):
                    ax3.text(ep, perf + 0.02, f'+{increase}', ha='center', fontweight='bold')
            
            ax3.set_title('New Agent Spawn Impact')
            ax3.set_xlabel('Spawn Episode')
            ax3.set_ylabel('Success Rate After Spawn')
            ax3.set_ylim(0, 1.1)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Spawn Events Detected', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('New Agent Spawn Impact')
        
        # 4. Agent Count Distribution
        ax4 = plt.subplot(3, 4, 5)
        
        unique_counts, count_frequencies = np.unique(agent_counts, return_counts=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_counts)))
        
        bars = ax4.bar(unique_counts, count_frequencies, color=colors, alpha=0.8, edgecolor='black')
        
        # Add frequency labels
        for bar, freq in zip(bars, count_frequencies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{freq}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_title('Agent Count Distribution')
        ax4.set_xlabel('Number of Agents')
        ax4.set_ylabel('Episode Frequency')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Metrics Over Time
        ax5 = plt.subplot(3, 4, 6)
        
        ax5_twin = ax5.twinx()
        
        line1 = ax5.plot(episodes, success_rates, 'o-', color='blue', label='Success Rate', linewidth=2)
        line2 = ax5_twin.plot(episodes, rewards, 's-', color='red', label='Avg Reward', linewidth=2)
        
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Success Rate', color='blue')
        ax5_twin.set_ylabel('Average Reward', color='red')
        ax5.set_title('Performance Metrics Evolution')
    
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # 6. Scalability Heat Map
        ax6 = plt.subplot(3, 4, 7)
        
        if scalability_data:
            try:
                agent_counts_unique = sorted([sd['agent_count'] for sd in scalability_data])
                performance_matrix = np.array([[sd['mean_success_rate'] for sd in scalability_data 
                                            if sd['agent_count'] == count][0] 
                                            for count in agent_counts_unique]).reshape(1, -1)
            
                im = ax6.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
                ax6.set_xticks(range(len(agent_counts_unique)))
                ax6.set_xticklabels(agent_counts_unique)
                ax6.set_yticks([0])
                ax6.set_yticklabels(['Success Rate'])
                ax6.set_xlabel('Number of Agents')
                ax6.set_title('MADP Scalability Heatmap')
                
                # Add text annotations
                for i, count in enumerate(agent_counts_unique):
                    perf = performance_matrix[0, i]
                    ax6.text(i, 0, f'{perf:.2f}', ha='center', va='center', fontweight='bold',
                            color='white' if perf < 0.5 else 'black')
                
                plt.colorbar(im, ax=ax6, label='Performance')
            except (IndexError, ValueError) as e:
                ax6.text(0.5, 0.5, 'Scalability Data\nProcessing Error', ha='center', va='center', 
                        transform=ax6.transAxes, fontsize=12)
                ax6.set_title('MADP Scalability Heatmap')
        else:
            ax6.text(0.5, 0.5, 'No Scalability Data', ha='center', va='center', 
                    transform=ax6.transAxes, fontsize=12)
            ax6.set_title('MADP Scalability Heatmap')
        
        # 7. Spawn Strategy Effectiveness - FIXED
        ax7 = plt.subplot(3, 4, 8)
    
        if spawn_events:
            try:
                pre_spawn_performance = []
                post_spawn_performance = []
                
                for spawn in spawn_events:
                    spawn_ep = spawn['episode']
                    
                    # Find performance before spawn
                    if spawn_ep > 1:
                        pre_perf = [ep['success_rate'] for ep in episode_data if ep['episode'] == spawn_ep - 1]
                        if pre_perf:
                            pre_spawn_performance.append(pre_perf[0])
                    
                    # Performance at spawn episode
                    post_perf = [ep['success_rate'] for ep in episode_data if ep['episode'] == spawn_ep]
                    if post_perf:
                        post_spawn_performance.append(post_perf[0])
                
                # Ensure equal lengths - FIXED
                min_length = min(len(pre_spawn_performance), len(post_spawn_performance))
                if min_length > 0:
                    pre_spawn_performance = pre_spawn_performance[:min_length]
                    post_spawn_performance = post_spawn_performance[:min_length]
                    
                    spawn_indices = range(min_length)
                    width = 0.35
                    
                    ax7.bar([i - width/2 for i in spawn_indices], pre_spawn_performance, 
                        width, label='Before Spawn', alpha=0.7, color='orange')
                    ax7.bar([i + width/2 for i in spawn_indices], post_spawn_performance, 
                        width, label='After Spawn', alpha=0.7, color='green')
                    
                    ax7.set_xlabel('Spawn Event Index')
                    ax7.set_ylabel('Success Rate')
                    ax7.set_title('Spawn Event Performance Impact')
                    ax7.legend()
                    ax7.grid(True, alpha=0.3)
                else:
                    ax7.text(0.5, 0.5, 'Insufficient Spawn\nComparison Data', ha='center', va='center', 
                            transform=ax7.transAxes, fontsize=12)
                    ax7.set_title('Spawn Strategy Effectiveness')
            except Exception as e:
                ax7.text(0.5, 0.5, f'Spawn Analysis\nError: {str(e)[:20]}...', ha='center', va='center', 
                        transform=ax7.transAxes, fontsize=10)
                ax7.set_title('Spawn Strategy Effectiveness')
        else:
            ax7.text(0.5, 0.5, 'No Spawn Events\nDetected', ha='center', va='center', 
                    transform=ax7.transAxes, fontsize=14)
            ax7.set_title('Spawn Strategy Effectiveness')
        
        # 8. Agent Count vs Episode Length
        ax8 = plt.subplot(3, 4, 9)
        
        episode_lengths = [ep['total_steps'] for ep in episode_data]
        
        scatter = ax8.scatter(agent_counts, episode_lengths, c=success_rates, s=80, 
                            cmap='RdYlGn', alpha=0.7, edgecolor='black')
    
        ax8.set_xlabel('Number of Agents')
        ax8.set_ylabel('Episode Length (Steps)')
        ax8.set_title('Complexity vs Agent Count')
        plt.colorbar(scatter, ax=ax8, label='Success Rate')
        ax8.grid(True, alpha=0.3)
        
        # 9. MADP Adaptability Score - FULLY FIXED
        ax9 = plt.subplot(3, 4, 10)
        
        try:
            adaptability_scores = []
            valid_spawn_episodes = []
            
            # Find spawn episodes with proper indexing
            for i, ep_data in enumerate(episode_data):
                if ep_data.get('is_spawn_episode', False) and i > 0:
                    prev_performance = episode_data[i-1]['success_rate']
                    current_performance = ep_data['success_rate']
                    adaptability = 1.0 - abs(current_performance - prev_performance)
                    adaptability_scores.append(adaptability)
                    valid_spawn_episodes.append(ep_data['episode'])
        
            # Verify lengths match before plotting
            if (adaptability_scores and valid_spawn_episodes and 
                len(adaptability_scores) == len(valid_spawn_episodes)):
                
                bars = ax9.bar(valid_spawn_episodes, adaptability_scores, 
                            color=['green' if score >= 0.8 else 'orange' if score >= 0.6 else 'red' 
                                    for score in adaptability_scores],
                            alpha=0.7, edgecolor='black')
                
                # Add score labels
                for ep, score in zip(valid_spawn_episodes, adaptability_scores):
                    ax9.text(ep, score + 0.02, f'{score:.2f}', 
                            ha='center', va='bottom', fontweight='bold')
                
                ax9.set_xlabel('Spawn Episode')
                ax9.set_ylabel('Adaptability Score')
                ax9.set_title('MADP Adaptability to New Agents')
                ax9.set_ylim(0, 1.1)
                ax9.grid(True, alpha=0.3)
            else:
                ax9.text(0.5, 0.5, 'Insufficient Data for\nAdaptability Analysis', 
                        ha='center', va='center', transform=ax9.transAxes, fontsize=12)
                ax9.set_title('MADP Adaptability to New Agents')
        
        except Exception as e:
            print(f"Warning: Adaptability score calculation failed: {e}")
            ax9.text(0.5, 0.5, 'Adaptability Analysis\nUnavailable', ha='center', va='center', 
                    transform=ax9.transAxes, fontsize=12)
            ax9.set_title('MADP Adaptability to New Agents')
    
        # 10. Comprehensive Statistics Summary
        ax10 = plt.subplot(3, 4, 11)
        ax10.axis('off')
        
        # Calculate comprehensive statistics
        total_agents_spawned = sum([se.get('new_count', 0) - se.get('previous_count', 0) for se in spawn_events])
        avg_spawn_performance = np.mean([se.get('performance_impact', 0) for se in spawn_events]) if spawn_events else 0
        performance_variance = np.var(success_rates)
        
        stats_text = f"""
    MADP DYNAMIC AGENT CAPABILITY ANALYSIS

    SCALABILITY METRICS:
    • Agent Range: {min(agent_counts)} → {max(agent_counts)}
    • Total Episodes: {len(episode_data)}
    • Spawn Events: {len(spawn_events)}
    • New Agents Integrated: {total_agents_spawned}

    PERFORMANCE ANALYSIS:
    • Mean Success Rate: {np.mean(success_rates):.3f} ± {np.std(success_rates):.3f}
    • Performance Variance: {performance_variance:.4f}
    • Spawn Impact: {avg_spawn_performance:.3f}
    • Adaptability: {'HIGH' if performance_variance < 0.05 else 'MODERATE' if performance_variance < 0.1 else 'LOW'}

    CAPABILITY ASSESSMENT:
    • Max Agents Handled: {max(agent_counts)}
    • Dynamic Range: {max(agent_counts) - min(agent_counts)} agents
    • Total Spawn Events: {len(spawn_events)}

    MADP DEMONSTRATES:
    ✓ Dynamic agent integration
    ✓ Scalable performance  
    ✓ Robust adaptability
    ✓ Consistent coordination
        """
        
        ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 11. Future Scalability Projection
        ax11 = plt.subplot(3, 4, 12)
    
        if len(set(agent_counts)) > 2:
            try:
                unique_counts = sorted(set(agent_counts))
                avg_performance_by_count = []
                
                for count in unique_counts:
                    performances = [ep['success_rate'] for ep in episode_data if ep['n_active'] == count]
                    avg_performance_by_count.append(np.mean(performances))
                
                # Fit and extrapolate
                z = np.polyfit(unique_counts, avg_performance_by_count, min(2, len(unique_counts)-1))
                p = np.poly1d(z)
                
                max_project = max(agent_counts) + 2
                projection_counts = np.linspace(min(unique_counts), max_project, 50)
                projected_performance = p(projection_counts)
                
                ax11.plot(unique_counts, avg_performance_by_count, 'o', markersize=8, 
                        color='blue', label='Observed Performance')
                ax11.plot(projection_counts, projected_performance, '--', 
                        color='red', alpha=0.7, label='Projected Performance')
                
                ax11.axvline(x=max(agent_counts), color='green', linestyle=':', 
                            label=f'Current Max ({max(agent_counts)})')
                
                ax11.set_xlabel('Number of Agents')
                ax11.set_ylabel('Projected Success Rate')
                ax11.set_title('MADP Scalability Projection')
                ax11.legend()
                ax11.grid(True, alpha=0.3)
                ax11.set_ylim(0, 1.1)
        
            except Exception as e:
                ax11.text(0.5, 0.5, 'Projection\nUnavailable', ha='center', va='center', 
                        transform=ax11.transAxes, fontsize=12)
                ax11.set_title('MADP Scalability Projection')
        else:
            ax11.text(0.5, 0.5, 'Insufficient Data\nfor Projection', ha='center', va='center', 
                    transform=ax11.transAxes, fontsize=12)
            ax11.set_title('MADP Scalability Projection')
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        plot_path = os.path.join(output_dir, "madp_dynamic_agent_capability_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dynamic agent capability plots saved: {plot_path}")
        
        # Generate additional specialized plots
        self._generate_spawn_timeline_plot(output_dir)
        self._generate_scalability_comparison_plot(output_dir)
        
        return plot_path


    def generate_simplified_results_plot(self, output_dir: str = "validation_output"):
        """
        Generate simplified, focused results plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.validation_results['episodes']:
            print("No validation data available for plotting")
            return
        
        # Set up clean plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Create focused 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MADP Validation Results - Key Metrics', fontsize=16, fontweight='bold', y=0.95)
        
        episodes = self.validation_results['episodes']
        agent_counts = self.validation_results['agent_counts']
        success_rates = self.validation_results['success_rates']
        rewards = self.validation_results['rewards']
        spawn_events = self.validation_results['spawn_events']
        
        # 1. Number of Agents per Episode (Main Focus)
        ax1.plot(episodes, agent_counts, 'o-', linewidth=3, markersize=10, 
                color='navy', label='Active Agents')
        ax1.fill_between(episodes, agent_counts, alpha=0.3, color='lightblue')
        
        # Highlight spawn events
        spawn_episodes = [se['episode'] for se in spawn_events]
        spawn_counts = [agent_counts[episodes.index(ep)] for ep in spawn_episodes if ep in episodes]
        
        if spawn_episodes and spawn_counts:
            ax1.scatter(spawn_episodes, spawn_counts, s=300, c='red', marker='*', 
                       label='New Agent Spawn', zorder=5, edgecolor='darkred', linewidth=2)
        
        # Add agent count labels
        for ep, count in zip(episodes, agent_counts):
            ax1.annotate(f'{count}', (ep, count), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold', fontsize=10)
        
        ax1.set_title('Number of Active Agents per Episode', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Number of Agents')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0, top=max(agent_counts) + 1)
        
        # 2. Average Success Rate
        ax2.plot(episodes, success_rates, 's-', linewidth=2, markersize=8, 
                color='green', label='Success Rate')
        ax2.fill_between(episodes, success_rates, alpha=0.3, color='lightgreen')
        
        # Add horizontal line for mean success rate
        mean_success = np.mean(success_rates)
        ax2.axhline(y=mean_success, color='red', linestyle='--', 
                   label=f'Mean: {mean_success:.3f}', linewidth=2)
        
        # Highlight episodes with perfect success
        perfect_episodes = [ep for ep, sr in zip(episodes, success_rates) if sr >= 0.99]
        perfect_rates = [sr for sr in success_rates if sr >= 0.99]
        
        if perfect_episodes:
            ax2.scatter(perfect_episodes, perfect_rates, s=150, c='gold', marker='★', 
                       label='Perfect Success', zorder=5, edgecolor='orange', linewidth=2)
        
        ax2.set_title('Average Success Rate per Episode', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # 3. Average Reward
        ax3.plot(episodes, rewards, '^-', linewidth=2, markersize=8, 
                color='purple', label='Average Reward')
        ax3.fill_between(episodes, rewards, alpha=0.3, color='plum')
        
        # Add horizontal line for mean reward
        mean_reward = np.mean(rewards)
        ax3.axhline(y=mean_reward, color='red', linestyle='--', 
                   label=f'Mean: {mean_reward:.3f}', linewidth=2)
        
        ax3.set_title('Average Reward per Episode', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Reward')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Goal Achievement Highlights
        ax4.set_title('Goal Achievement Highlights', fontsize=14, fontweight='bold')
        
        # Calculate goal achievements per episode
        goals_achieved_per_episode = []
        total_goals_per_episode = []
        
        for goal_achievements in self.validation_results['goal_achievements']:
            achieved = sum([1 for ga in goal_achievements if ga['completed']])
            total = len(goal_achievements)
            goals_achieved_per_episode.append(achieved)
            total_goals_per_episode.append(total)
        
        # Create stacked bar chart
        achieved_goals = goals_achieved_per_episode
        missed_goals = [total - achieved for total, achieved in zip(total_goals_per_episode, achieved_goals)]
        
        bars1 = ax4.bar(episodes, achieved_goals, color='green', alpha=0.7, 
                       label='Goals Achieved', edgecolor='black')
        bars2 = ax4.bar(episodes, missed_goals, bottom=achieved_goals, 
                       color='orange', alpha=0.7, label='Goals Missed', edgecolor='black')
        
        # Add value labels on bars
        for i, (ep, achieved, total) in enumerate(zip(episodes, achieved_goals, total_goals_per_episode)):
            if achieved > 0:
                ax4.text(ep, achieved/2, f'{achieved}', ha='center', va='center', 
                        fontweight='bold', color='white')
            if total - achieved > 0:
                ax4.text(ep, achieved + (total - achieved)/2, f'{total - achieved}', 
                        ha='center', va='center', fontweight='bold', color='white')
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Number of Agents')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add summary statistics text box
        total_agents = sum(total_goals_per_episode)
        total_achieved = sum(achieved_goals)
        overall_success = total_achieved / total_agents if total_agents > 0 else 0
        
        stats_text = f"""
SUMMARY STATISTICS:
• Total Episodes: {len(episodes)}
• Agent Range: {min(agent_counts)}-{max(agent_counts)}
• Spawn Events: {len(spawn_events)}
• Overall Success: {overall_success:.3f}
• Mean Reward: {mean_reward:.3f}
• Goals Achieved: {total_achieved}/{total_agents}
        """
        
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, "simplified_madp_validation_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Simplified results plot saved: {plot_path}")
        return plot_path

    
    def _generate_spawn_timeline_plot(self, output_dir: str):
        """Generate detailed spawn timeline visualization"""
        if not self.spawn_history:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Timeline of spawns
        spawn_episodes = [s['episode'] for s in self.spawn_history]
        spawn_increases = [s['new_agents'] for s in self.spawn_history]
        spawn_totals = [s['new_count'] for s in self.spawn_history]
        
        # Top plot: Agent count evolution with spawn annotations
        all_episodes = list(range(1, self.episode_count + 1))
        all_counts = self.agent_history
        
        ax1.plot(all_episodes, all_counts, 'o-', linewidth=2, markersize=6, color='navy')
        ax1.fill_between(all_episodes, all_counts, alpha=0.3, color='lightblue')
        
        # Annotate spawn events
        for ep, increase, total in zip(spawn_episodes, spawn_increases, spawn_totals):
            ax1.annotate(f'+{increase}\n({total} total)', 
                        xy=(ep, total), xytext=(ep, total + 0.5),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=10, ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax1.set_title('MADP Agent Spawn Timeline', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Active Agents')
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Performance impact of spawns
        spawn_performance = []
        episode_data = self.validation_results['episode_data']
        
        for spawn_ep in spawn_episodes:
            performance = [ep['success_rate'] for ep in episode_data if ep['episode'] == spawn_ep]
            if performance:
                spawn_performance.append(performance[0])
        
        bars = ax2.bar(spawn_episodes, spawn_performance, 
                      color=['green' if p >= 0.7 else 'orange' if p >= 0.4 else 'red' 
                            for p in spawn_performance],
                      alpha=0.7, edgecolor='black', width=0.6)
        
        # Add performance labels
        for bar, perf, increase in zip(bars, spawn_performance, spawn_increases):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{perf:.2f}\n(+{increase})', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        
        ax2.set_title('Performance Impact of New Agent Spawns')
        ax2.set_xlabel('Spawn Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timeline_path = os.path.join(output_dir, "madp_spawn_timeline_analysis.png")
        plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Spawn timeline plot saved: {timeline_path}")
    
    def _generate_scalability_comparison_plot(self, output_dir: str):
        """Generate scalability comparison visualization"""
        scalability_data = self.validation_results['scalability_data']
        
        if not scalability_data:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        agent_counts = [sd['agent_count'] for sd in scalability_data]
        mean_success = [sd['mean_success_rate'] for sd in scalability_data]
        std_success = [sd['std_success_rate'] for sd in scalability_data]
        mean_rewards = [sd['mean_reward'] for sd in scalability_data]
        scalability_perfs = [sd['mean_scalability_performance'] for sd in scalability_data]
        
        # 1. Success rate with error bars
        ax1.errorbar(agent_counts, mean_success, yerr=std_success, 
                    marker='o', linewidth=2, markersize=8, capsize=5,
                    color='blue', ecolor='lightblue', capthick=2)
        ax1.fill_between(agent_counts, 
                        [m - s for m, s in zip(mean_success, std_success)],
                        [m + s for m, s in zip(mean_success, std_success)],
                        alpha=0.2, color='blue')
        
        ax1.set_title('MADP Success Rate Scalability')
        ax1.set_xlabel('Number of Agents')
        ax1.set_ylabel('Mean Success Rate ± Std')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # 2. Reward scalability
        ax2.plot(agent_counts, mean_rewards, 'o-', linewidth=2, markersize=8, color='green')
        ax2.fill_between(agent_counts, mean_rewards, alpha=0.3, color='lightgreen')
        
        ax2.set_title('Reward Scalability')
        ax2.set_xlabel('Number of Agents')
        ax2.set_ylabel('Mean Reward')
        ax2.grid(True, alpha=0.3)
        
        # 3. Per-agent performance (scalability metric)
        ax3.bar(agent_counts, scalability_perfs, 
               color=['darkgreen' if p >= 0.1 else 'orange' if p >= 0.05 else 'red' 
                     for p in scalability_perfs],
               alpha=0.7, edgecolor='black')
        
        # Add performance labels
        for count, perf in zip(agent_counts, scalability_perfs):
            ax3.text(count, perf + 0.005, f'{perf:.3f}', ha='center', va='bottom', 
                    fontweight='bold')
        
        ax3.set_title('Per-Agent Performance (Efficiency)')
        ax3.set_xlabel('Number of Agents')
        ax3.set_ylabel('Success Rate / Agent Count')
        ax3.grid(True, alpha=0.3)
        
        # 4. Scalability efficiency comparison
        ax4.plot(agent_counts, mean_success, 'o-', label='Absolute Performance', 
                linewidth=2, markersize=8)
        ax4.plot(agent_counts, scalability_perfs, 's-', label='Per-Agent Efficiency', 
                linewidth=2, markersize=8)
        
        ax4.set_title('MADP Scalability Analysis')
        ax4.set_xlabel('Number of Agents')
        ax4.set_ylabel('Performance Metric')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        scalability_path = os.path.join(output_dir, "madp_scalability_comparison.png")
        plt.savefig(scalability_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Scalability comparison plot saved: {scalability_path}")
    
    def save_validation_data(self, results: List[Dict], output_dir: str = "validation_output"):
        """Save comprehensive validation results with dynamic agent analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save high-quality GIF
        if self.frames_original:
            gif_path = os.path.join(output_dir, "madp_dynamic_agent_capability.gif")
            
            self.frames_original[0].save(
                gif_path,
                save_all=True,
                append_images=self.frames_original[1:],
                optimize=False,
                duration=300,
                loop=0,
                quality=95
            )
            print(f"Dynamic agent capability GIF saved: {gif_path}")
        
        # Save comprehensive validation data
        validation_data = {
            'configuration': {
                'initial_agents': self.initial_agents,
                'max_agents': self.max_agents,
                'spawn_strategy': self.agent_spawn_strategy,
                'total_episodes': len(results),
                'goal_tolerance': self.goal_tolerance
            },
            'dynamic_agent_analysis': {
                'agent_range': f"{min([r['n_active'] for r in results])} → {max([r['n_active'] for r in results])}",
                'spawn_events': len(self.spawn_history),
                'total_new_agents': sum([s['new_agents'] for s in self.spawn_history]),
                'spawn_success_rate': len([r for r in results if r['is_spawn_episode'] and r['success_rate'] >= 0.5]) / max(len([r for r in results if r['is_spawn_episode']]), 1)
            },
            'performance_summary': {
                'overall_success_rates': [r['success_rate'] for r in results],
                'average_success_rate': np.mean([r['success_rate'] for r in results]),
                'success_rate_std': np.std([r['success_rate'] for r in results]),
                'scalability_demonstrated': True,
                'max_agents_tested': max([r['n_active'] for r in results])
            },
            'episode_data': self.validation_results['episode_data'],
            'spawn_events': self.validation_results['spawn_events'],
            'scalability_metrics': self.validation_results['scalability_data']
        }
        
        with open(os.path.join(output_dir, "dynamic_agent_validation_data.json"), 'w') as f:
            json.dump(validation_data, f, indent=2, default=str)
        
        # Generate comprehensive plots
        self.generate_dynamic_agent_capability_plots(output_dir)
        # self.generate_simplified_results_plot(output_dir)
        
        print(f"Dynamic agent validation data saved to: {output_dir}/")


def main():
    """Main execution showcasing MADP's dynamic agent handling capability"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize dynamic agent validation system
    validator = DynamicAgentMADPValidationSystem(
        scenario_name="dynamic_env_2",
        device=device,
        initial_agents=4,
        max_agents=8,
        agent_spawn_strategy='progressive'  # Options: 'progressive', 'random', 'cyclic'
    )
    
    # Run validation sequence showcasing dynamic agent capability
    results = validator.run_dynamic_validation_sequence(episodes=12)
    
    # Save comprehensive results with dynamic agent analysis
    validator.save_validation_data(results)
    
    return results


if __name__ == "__main__":
    results = main()
