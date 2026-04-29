"""
Dynamic Multi-Agent Navigation Scenario for MADP Demonstration
Enhanced version of navigation_v2 with dynamic agent spawning/removal (3-8 agents)
"""

import typing
from typing import Callable, Dict, List
import torch
from torch import Tensor
import numpy as np
import random

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World, Line, Box
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

class DynamicNavigationScenario(BaseScenario):
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Dynamic agent configuration
        self.min_agents = kwargs.pop("min_agents", 3)
        self.max_agents = kwargs.pop("max_agents", 8)
        self.spawn_probability = kwargs.pop("spawn_probability", 0.02)
        self.despawn_on_goal = kwargs.pop("despawn_on_goal", True)
        
        # Environment configuration
        self.plot_grid = kwargs.pop("plot_grid", False)
        self.collisions = kwargs.pop("collisions", True)
        self.lidar_range = kwargs.pop("lidar_range", 0.15)
        self.agent_radius = kwargs.pop("agent_radius", 0.05)
        self.comms_range = kwargs.pop("comms_range", 0.3)
        
        # Reward configuration
        self.shared_rew = kwargs.pop("shared_rew", False)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1.0)
        self.final_reward = kwargs.pop("final_reward", 10.0)
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -2.0)
        self.goal_completion_bonus = kwargs.pop("goal_completion_bonus", 15.0)
        
        # World parameters
        self.world_semidim = kwargs.pop("world_semidim", 1.5)
        self.floor_width = kwargs.pop("floor_width", 3.0)
        self.floor_length = kwargs.pop("floor_length", 3.0)
        self.min_distance_between_entities = self.agent_radius * 2 + 0.08
        self.min_collision_distance = self.agent_radius * 2
        self.goal_threshold = kwargs.pop("goal_threshold", 0.08)
        
        # Rendering configuration
        self.viewer_size = (1000, 800)
        self.viewer_zoom = 1.2
        self.render_origin = (0.0, 0.0)
        
        ScenarioUtils.check_kwargs_consumed(kwargs)
        
        # Dynamic state tracking
        self.current_active_agents = self.min_agents
        self.active_agent_indices = list(range(self.min_agents))
        self.spawn_timer = 0
        self.completion_times = {}
        self.agent_spawn_history = []
        
        # Create world
        world = World(batch_dim, device, substeps=5)
        
        # Agent colors for visual distinction
        self.agent_colors = [
            Color.BLUE.value,      # Initial agents
            Color.GREEN.value,     # Spawned agents
            Color.ORANGE.value,    # Dynamic agents
            Color.PURPLE.value,    # Late spawned agents
            Color.CYAN.value,      # Additional agents
            Color.MAGENTA.value,   # Extra agents
            Color.YELLOW.value,    # Maximum capacity
            Color.RED.value,       # Final agents
        ]
        
        # Create agent pool (maximum capacity)
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent) and not getattr(e, 'finished', False)
        
        for i in range(self.max_agents):
            agent_color = self.agent_colors[min(i, len(self.agent_colors) - 1)]
            
            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=agent_color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=(
                    Lidar(
                        world,
                        n_rays=20,
                        max_range=self.lidar_range,
                        entity_filter=entity_filter_agents,
                    ),
                ) if self.collisions else (None,),
            )
            
            # Initialize agent tracking variables
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = torch.zeros(batch_dim, device=device)
            agent.goal_completion_rew = torch.zeros(batch_dim, device=device)
            agent.distance_to_goal = torch.full((batch_dim,), float('inf'), device=device)
            agent.on_goal = torch.zeros(batch_dim, dtype=torch.bool, device=device)
            agent.pos_shaping = torch.zeros(batch_dim, device=device)
            agent.finished = torch.zeros(batch_dim, dtype=torch.bool, device=device)
            agent.spawn_time = torch.zeros(batch_dim, device=device)
            agent.active = i < self.min_agents  # Initially only min_agents are active
            
            world.add_agent(agent)
        
        # Create goal landmarks
        goal_colors = [
            Color.LIGHT_GREEN.value,
            Color.GOLD.value,
            Color.CORAL.value,
            Color.PINK.value,
        ]
        
        for i in range(self.max_agents):
            goal_color = goal_colors[min(i // 2, len(goal_colors) - 1)]
            
            goal = Landmark(
                name=f"goal_{i}",
                collide=False,
                color=goal_color,
                shape=Sphere(radius=self.goal_threshold),
            )
            world.add_landmark(goal)
            world.agents[i].goal = goal
        
        # Initialize boundary walls
        self._init_boundary_walls(world)
        
        # Global reward tracking
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = torch.zeros(batch_dim, device=device)
        self.completion_rew = torch.zeros(batch_dim, device=device)
        
        return world
    
    def _init_boundary_walls(self, world):
        """Initialize boundary walls for the environment"""
        wall_configs = [
            {"name": "Top Wall", "pos": [0, self.floor_width / 2], "rot": 0},
            {"name": "Bottom Wall", "pos": [0, -self.floor_width / 2], "rot": 0},
            {"name": "Left Wall", "pos": [-self.floor_length / 2, 0], "rot": torch.pi / 2},
            {"name": "Right Wall", "pos": [self.floor_length / 2, 0], "rot": torch.pi / 2},
        ]
        
        for config in wall_configs:
            wall = Landmark(
                name=config["name"],
                collide=True,
                movable=False,
                shape=Line(length=max(self.floor_width, self.floor_length)),
                color=Color.BLACK,
            )
            world.add_landmark(wall)
    
    def reset_world_at(self, env_index: int = None):
        """Reset world state with initial active agents"""
        # Reset dynamic state
        self.current_active_agents = self.min_agents
        self.active_agent_indices = list(range(self.min_agents))
        self.spawn_timer = 0
        self.completion_times.clear()
        self.agent_spawn_history.clear()
        
        # Reset agent states
        for i, agent in enumerate(self.world.agents):
            if i < self.min_agents:  # Active agents
                agent.active = True
                agent.finished.fill_(False)
                agent.spawn_time.fill_(0)
            else:  # Inactive agents - position off-screen
                agent.active = False
                agent.finished.fill_(True)
                off_screen_pos = torch.tensor([100.0, 100.0], device=agent.state.pos.device)
                if env_index is None:
                    agent.state.pos.fill_(0).add_(off_screen_pos)
                else:
                    agent.state.pos[env_index] = off_screen_pos
        
        # Spawn active agents randomly
        active_agents = [agent for agent in self.world.agents if agent.active]
        ScenarioUtils.spawn_entities_randomly(
            active_agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            x_bounds=(-self.world_semidim * 0.8, self.world_semidim * 0.8),
            y_bounds=(-self.world_semidim * 0.8, self.world_semidim * 0.8),
        )
        
        # Position goals for active agents
        occupied_positions = torch.stack([agent.state.pos for agent in active_agents], dim=1)
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)
        
        for i, agent in enumerate(self.world.agents):
            if agent.active:
                # Find valid goal position
                goal_position = ScenarioUtils.find_random_pos_for_entity(
                    occupied_positions=occupied_positions,
                    env_index=env_index,
                    world=self.world,
                    min_dist_between_entities=self.min_distance_between_entities * 2,
                    x_bounds=(-self.world_semidim * 0.9, self.world_semidim * 0.9),
                    y_bounds=(-self.world_semidim * 0.9, self.world_semidim * 0.9),
                )
                agent.goal.set_pos(goal_position.squeeze(1), batch_index=env_index)
                occupied_positions = torch.cat([occupied_positions, goal_position], dim=1)
                
                # Initialize position shaping
                if env_index is None:
                    agent.pos_shaping = (
                        torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=1)
                        * self.pos_shaping_factor
                    )
                else:
                    agent.pos_shaping[env_index] = (
                        torch.linalg.vector_norm(
                            agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                        ) * self.pos_shaping_factor
                    )
            else:
                # Position inactive goals off-screen
                off_screen_pos = torch.tensor([100.0, 100.0], device=agent.goal.state.pos.device)
                if env_index is None:
                    agent.goal.state.pos.fill_(0).add_(off_screen_pos)
                else:
                    agent.goal.state.pos[env_index] = off_screen_pos
        
        # Reset boundary walls
        self._reset_boundary_walls(env_index)
    
    def _reset_boundary_walls(self, env_index: int = None):
        """Reset positions of boundary walls"""
        wall_configs = {
            "Top Wall": {"pos": [0, self.floor_width / 2], "rot": 0},
            "Bottom Wall": {"pos": [0, -self.floor_width / 2], "rot": 0},
            "Left Wall": {"pos": [-self.floor_length / 2, 0], "rot": torch.pi / 2},
            "Right Wall": {"pos": [self.floor_length / 2, 0], "rot": torch.pi / 2},
        }
        
        for landmark in self.world.landmarks:
            if landmark.name in wall_configs:
                config = wall_configs[landmark.name]
                pos_tensor = torch.tensor(config["pos"], dtype=torch.float32, device=self.world.device)
                rot_tensor = torch.tensor([config["rot"]], dtype=torch.float32, device=self.world.device)
                
                landmark.set_pos(pos_tensor, batch_index=env_index)
                landmark.set_rot(rot_tensor, batch_index=env_index)
    
    def reward(self, agent: Agent):
        """Compute reward for individual agent with dynamic considerations"""
        agent_idx = int(agent.name.split('_')[1])
        is_first = agent == self.world.agents[0]
        
        # Global reward computation
        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0
            self.completion_rew[:] = 0
            
            # Compute rewards for all active agents
            for a in self.world.agents:
                if a.active:
                    self.pos_rew += self._compute_agent_reward(a)
                a.agent_collision_rew[:] = 0
                a.goal_completion_rew[:] = 0
            
            # Check completion status
            active_agents = [a for a in self.world.agents if a.active]
            if active_agents:
                self.all_goal_reached = torch.all(
                    torch.stack([a.on_goal for a in active_agents], dim=-1),
                    dim=-1,
                )
                self.final_rew[self.all_goal_reached] += self.final_reward
            
            # Handle collisions between active agents
            for i, a in enumerate(self.world.agents):
                if not a.active:
                    continue
                for j, b in enumerate(self.world.agents):
                    if j <= i or not b.active:
                        continue
                    
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        collision_mask = distance <= self.min_collision_distance
                        
                        a.agent_collision_rew[collision_mask] += self.agent_collision_penalty
                        b.agent_collision_rew[collision_mask] += self.agent_collision_penalty
        
        # Return appropriate reward based on agent status
        if not agent.active:
            return torch.zeros_like(agent.pos_rew)
        
        # Goal completion bonus
        newly_completed = agent.on_goal & ~getattr(agent, '_was_on_goal', torch.zeros_like(agent.on_goal))
        agent.goal_completion_rew[newly_completed] += self.goal_completion_bonus
        agent._was_on_goal = agent.on_goal.clone()
        
        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew + agent.goal_completion_rew
    
    def _compute_agent_reward(self, agent: Agent):
        """Compute position-based reward for individual agent"""
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos, dim=-1
        )
        agent.on_goal = agent.distance_to_goal < self.goal_threshold
        
        # Position shaping reward
        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        
        return agent.pos_rew
    
    def observation(self, agent: Agent):
        """Generate observation for agent including dynamic environment state"""
        agent_idx = int(agent.name.split('_')[1])
        
        if not agent.active:
            # Return zero observation for inactive agents
            obs_size = self._get_observation_size()
            return torch.zeros(agent.state.pos.shape[0], obs_size, device=agent.state.pos.device)
        
        observations = [
            agent.state.pos,  # Own position
            agent.state.vel,  # Own velocity
            agent.goal.state.pos - agent.state.pos,  # Relative goal position
        ]
        
        # Other active agents' information
        for other in self.world.agents:
            if other != agent and other.active:
                relative_pos = other.state.pos - agent.state.pos
                distance = torch.linalg.vector_norm(relative_pos, dim=-1, keepdim=True)
                observations.extend([relative_pos, distance])
            elif other != agent:  # Inactive agents
                observations.extend([
                    torch.zeros_like(agent.state.pos),  # Zero relative position
                    torch.full_like(agent.state.pos[:, :1], 100.0)  # Large distance
                ])
        
        # Environment state information
        active_agent_count = torch.tensor(
            [len(self.active_agent_indices) / self.max_agents], 
            device=agent.state.pos.device
        ).expand(agent.state.pos.shape[0], 1)
        
        agent_id = torch.tensor(
            [agent_idx / self.max_agents], 
            device=agent.state.pos.device
        ).expand(agent.state.pos.shape[0], 1)
        
        observations.extend([active_agent_count, agent_id])
        
        # Lidar observations for collision avoidance
        if self.collisions and agent.sensors[0] is not None:
            lidar_obs = agent.sensors[0]._max_range - agent.sensors[0].measure()
            observations.append(lidar_obs)
        
        return torch.cat(observations, dim=-1)
    
    def _get_observation_size(self):
        """Calculate the size of observation vector"""
        base_size = 2 + 2 + 2  # pos + vel + relative_goal
        other_agents_size = (self.max_agents - 1) * 3  # relative_pos (2) + distance (1)
        env_state_size = 2  # active_count + agent_id
        lidar_size = 20 if self.collisions else 0  # Lidar rays
        
        return base_size + other_agents_size + env_state_size + lidar_size
    
    def step_scenario_dynamics(self):
        """Handle dynamic agent spawning and removal each step"""
        self.spawn_timer += 1
        
        # Handle goal completion and agent removal
        agents_to_remove = []
        for agent_idx in self.active_agent_indices[:]:
            agent = self.world.agents[agent_idx]
            
            if self.despawn_on_goal and torch.any(agent.on_goal):
                # Mark completed agents for removal
                completed_envs = agent.on_goal.nonzero().flatten()
                for env_idx in completed_envs:
                    if env_idx not in self.completion_times:
                        self.completion_times[env_idx.item()] = self.spawn_timer
                
                # Move agent off-screen
                agent.active = False
                agent.finished.fill_(True)
                off_screen_pos = torch.tensor([100.0, 100.0], device=agent.state.pos.device)
                agent.state.pos.fill_(0).add_(off_screen_pos)
                agent.goal.state.pos.fill_(0).add_(off_screen_pos)
                
                agents_to_remove.append(agent_idx)
        
        # Remove completed agents from active list
        for agent_idx in agents_to_remove:
            if agent_idx in self.active_agent_indices:
                self.active_agent_indices.remove(agent_idx)
                self.current_active_agents -= 1
        
        # Spawn new agents probabilistically
        if (self.current_active_agents < self.max_agents and 
            random.random() < self.spawn_probability):
            self._spawn_new_agent()
        
        # Ensure minimum number of agents
        while self.current_active_agents < self.min_agents:
            self._spawn_new_agent()
    
    def _spawn_new_agent(self):
        """Spawn a new agent in the environment"""
        available_indices = [
            i for i in range(self.max_agents) 
            if i not in self.active_agent_indices
        ]
        
        if not available_indices:
            return
        
        new_agent_idx = available_indices[0]
        agent = self.world.agents[new_agent_idx]
        
        # Find safe spawn position
        max_attempts = 100
        for attempt in range(max_attempts):
            spawn_pos = torch.tensor([
                np.random.uniform(-self.world_semidim * 0.7, self.world_semidim * 0.7),
                np.random.uniform(-self.world_semidim * 0.7, self.world_semidim * 0.7)
            ], device=agent.state.pos.device)
            
            # Check distance to active agents
            safe_spawn = True
            for active_idx in self.active_agent_indices:
                active_agent = self.world.agents[active_idx]
                distance = torch.linalg.vector_norm(spawn_pos - active_agent.state.pos, dim=-1)
                if torch.any(distance < self.min_distance_between_entities * 1.5):
                    safe_spawn = False
                    break
            
            if safe_spawn:
                break
        
        # Activate agent
        agent.active = True
        agent.finished.fill_(False)
        agent.spawn_time.fill_(self.spawn_timer)
        agent.state.pos.fill_(0).add_(spawn_pos)
        agent.state.vel.fill_(0)
        
        # Set goal position
        goal_pos = torch.tensor([
            np.random.uniform(-self.world_semidim * 0.8, self.world_semidim * 0.8),
            np.random.uniform(-self.world_semidim * 0.8, self.world_semidim * 0.8)
        ], device=agent.goal.state.pos.device)
        
        agent.goal.state.pos.fill_(0).add_(goal_pos)
        
        # Initialize position shaping
        agent.pos_shaping = (
            torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=-1)
            * self.pos_shaping_factor
        )
        
        # Add to active list
        self.active_agent_indices.append(new_agent_idx)
        self.current_active_agents += 1
        self.agent_spawn_history.append((self.spawn_timer, new_agent_idx))
    
    def done(self):
        """Check if scenario is complete - continues indefinitely for dynamic demo"""
        # For dynamic scenarios, we typically don't want to end
        # unless all agents have completed their goals multiple times
        return torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)
    
    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """Return comprehensive information about agent and environment state"""
        agent_idx = int(agent.name.split('_')[1])
        
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "completion_rew": agent.goal_completion_rew,
            "agent_collisions": agent.agent_collision_rew,
            "distance_to_goal": agent.distance_to_goal,
            "on_goal": agent.on_goal.float(),
            "active_agents_count": torch.tensor([self.current_active_agents], device=agent.state.pos.device).expand(agent.state.pos.shape[0]),
            "agent_active": torch.tensor([agent.active], device=agent.state.pos.device).expand(agent.state.pos.shape[0]),
            "spawn_time": agent.spawn_time,
            "scenario_step": torch.tensor([self.spawn_timer], device=agent.state.pos.device).expand(agent.state.pos.shape[0]),
        }
    
    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        """Additional rendering elements for dynamic visualization"""
        from vmas.simulator import rendering
        
        geoms: List["Geom"] = []
        
        # Draw communication links between active agents
        active_agents = [self.world.agents[i] for i in self.active_agent_indices]
        
        for i, agent1 in enumerate(active_agents):
            for j, agent2 in enumerate(active_agents):
                if j <= i:
                    continue
                
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                
                if agent_dist[env_index] <= self.comms_range:
                    line = rendering.Line(
                        agent1.state.pos[env_index],
                        agent2.state.pos[env_index],
                        width=2,
                    )
                    line.set_color(*Color.GRAY.value, alpha=0.4)
                    geoms.append(line)
        
        # Draw agent-goal connections
        for agent_idx in self.active_agent_indices:
            agent = self.world.agents[agent_idx]
            if agent.active:
                line = rendering.Line(
                    agent.state.pos[env_index],
                    agent.goal.state.pos[env_index],
                    width=1,
                )
                line.set_color(*Color.BLACK.value, alpha=0.3)
                geoms.append(line)
        
        return geoms


# Enhanced Heuristic Policy with Dynamic Agent Awareness
class DynamicCLFHeuristicPolicy(BaseHeuristicPolicy):
    
    def __init__(self, clf_epsilon=0.5, clf_slack=8.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf_epsilon = clf_epsilon
        self.clf_slack = clf_slack
    
    def compute_action(self, observation: Tensor, u_range: Tensor) -> Tensor:
        """Enhanced CLF-QP controller with dynamic agent considerations"""
        try:
            import cvxpy as cp
            from cvxpylayers.torch import CvxpyLayer
        except ImportError:
            # Fallback to simple proportional controller
            return self._proportional_controller(observation, u_range)
        
        batch_size = observation.shape[0]
        device = observation.device
        
        # Extract state information
        agent_pos = observation[:, :2]
        agent_vel = observation[:, 2:4]
        goal_relative = observation[:, 4:6]
        goal_pos = agent_pos + goal_relative
        
        # Enhanced Lyapunov function for dynamic environments
        pos_error = agent_pos - goal_pos
        V_value = (
            torch.sum(pos_error ** 2, dim=1) +
            0.5 * torch.sum(pos_error * agent_vel, dim=1) +
            torch.sum(agent_vel ** 2, dim=1)
        )
        
        LfV_val = (
            2 * torch.sum(pos_error * agent_vel, dim=1) +
            0.5 * torch.sum(agent_vel ** 2, dim=1)
        )
        
        LgV_vals = torch.stack([
            0.5 * pos_error[:, 0] + 2 * agent_vel[:, 0],
            0.5 * pos_error[:, 1] + 2 * agent_vel[:, 1],
        ], dim=1)
        
        # Solve QP for each environment in batch
        actions = []
        for env_idx in range(batch_size):
            # Define QP problem
            u = cp.Variable(2)
            V_param = cp.Parameter(1)
            lfV_param = cp.Parameter(1)
            lgV_params = cp.Parameter(2)
            clf_slack = cp.Variable(1)
            
            # Objective: minimize control effort + CLF slack
            objective = cp.Minimize(
                cp.sum_squares(u) + self.clf_slack * cp.sum_squares(clf_slack)
            )
            
            # Constraints
            constraints = [
                u <= u_range.item(),
                u >= -u_range.item(),
                lfV_param + lgV_params @ u + self.clf_epsilon * V_param + clf_slack >= 0
            ]
            
            # Create and solve problem
            problem = cp.Problem(objective, constraints)
            layer = CvxpyLayer(
                problem,
                parameters=[V_param, lfV_param, lgV_params],
                variables=[u, clf_slack]
            )
            
            try:
                solution = layer(
                    V_value[env_idx:env_idx+1],
                    LfV_val[env_idx:env_idx+1],
                    LgV_vals[env_idx:env_idx+1],
                    solver_args={"max_iters": 1000, "eps": 1e-4}
                )
                action = solution[0].squeeze(0)
                actions.append(torch.clamp(action, -u_range, u_range))
            except Exception:
                # Fallback to proportional control
                action = self._proportional_action(
                    goal_relative[env_idx], agent_vel[env_idx], u_range
                )
                actions.append(action)
        
        return torch.stack(actions)
    
    def _proportional_controller(self, observation: Tensor, u_range: Tensor) -> Tensor:
        """Fallback proportional controller for dynamic scenarios"""
        goal_relative = observation[:, 4:6]
        agent_vel = observation[:, 2:4]
        
        actions = []
        for env_idx in range(observation.shape[0]):
            action = self._proportional_action(
                goal_relative[env_idx], agent_vel[env_idx], u_range
            )
            actions.append(action)
        
        return torch.stack(actions)
    
    def _proportional_action(self, goal_relative: Tensor, agent_vel: Tensor, u_range: Tensor) -> Tensor:
        """Simple proportional control action"""
        kp_pos = 2.0
        kd_vel = 0.8
        
        action = kp_pos * goal_relative - kd_vel * agent_vel
        return torch.clamp(action, -u_range, u_range)


if __name__ == "__main__":
    # Demonstration of dynamic multi-agent navigation
    print("Dynamic Multi-Agent Navigation Scenario")
    print("Demonstrates MADP with varying agent counts (3-8)")
    
    scenario = DynamicNavigationScenario(
        min_agents=3,
        max_agents=8,
        spawn_probability=0.03,
        despawn_on_goal=True,
        world_semidim=1.8,
        final_reward=15.0,
        goal_completion_bonus=20.0,
    )
    
    render_interactively(
        scenario,
        control_two_agents=True,
        n_agents=8,  # Maximum capacity
    )
