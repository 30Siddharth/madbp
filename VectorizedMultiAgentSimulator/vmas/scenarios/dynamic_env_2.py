# Copyright (c) 2022-2024.
# ProrokLab (https://www.proroklab.org/)
# All rights reserved.

import typing
from typing import Callable, Dict, List
import torch
from torch import Tensor
import numpy as np

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World, Line, Box
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class DynamicNavigationScenario(BaseScenario):
    """
    Dynamic navigation scenario that creates only the required number of agents per episode.
    
    Key Features:
    - Creates exact number of agents needed per episode
    - Inactive agents are set to origin (0,0) or None positions
    - Only renders active agents
    - Supports state persistence across episodes
    """
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        
        # Agent count - use n_agents as the actual number to create
        self.n_agents = kwargs.pop("n_agents", 4)  # Number of agents to create
        self.current_active_agents = self.n_agents
        
        # Scenario parameters
        self.n_obs = kwargs.pop("n_obs", 0)
        self.collisions = kwargs.pop("collisions", True)
        self.agents_with_same_goal = kwargs.pop("agents_with_same_goal", 1)
        self.split_goals = kwargs.pop("split_goals", False)
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)
        self.lidar_range = kwargs.pop("lidar_range", 0.15)
        self.agent_radius = kwargs.pop("agent_radius", 0.05)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.shared_rew = kwargs.pop("shared_rew", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 0.01)
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
        
        # Environment parameters
        self.floor_width = kwargs.pop("width", 4)
        self.floor_length = kwargs.pop("length", 2)
        self.world_semidim = 1
        self.min_collision_distance = 0.05
        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        
        # Rendering parameters
        self.viewer_size = (800, 600)
        self.viewer_zoom = 1.5
        self.render_origin = (0.0, 0.0)
        
        # Test position parameters
        self.test_start_positions = kwargs.pop("test_start_positions", None)
        self.test_goal_positions = kwargs.pop("test_goal_positions", None)
        self.use_test_positions = kwargs.pop("use_test_positions", False)
        
        # Episode tracking for state persistence
        self.episode_count = 0
        self.agent_states_history = {}  # Store agent states between episodes
        self.goal_states_history = {}   # Store goal states between episodes
        self.agent_reached_goal = {}    # Track which agents reached their goals
        
        # Validation checks
        assert 1 <= self.agents_with_same_goal <= self.n_agents
        if self.agents_with_same_goal > 1:
            assert not self.collisions, "If agents share goals they cannot be collidable"
        
        if self.split_goals:
            assert (
                self.n_agents % 2 == 0 
                and self.agents_with_same_goal == self.n_agents // 2
            ), "Splitting goals requires even agents and half team same goal"
        
        ScenarioUtils.check_kwargs_consumed(kwargs)
        
        # Create world with substeps for physics
        world = World(batch_dim, device, substeps=2)
        
        # Color management
        known_colors = [color.value for color in Color]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )
        
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)
        
        # Create only the required number of agents
        for i in range(self.n_agents):
            color = (
                known_colors[i] 
                if i < len(known_colors) 
                else colors[i - len(known_colors)]
            )
            
            # Properly wrap sensor in list
            agent_sensors = []
            if self.collisions:
                agent_sensors.append(
                    Lidar(
                        world,
                        n_rays=20,
                        max_range=self.lidar_range,
                        entity_filter=entity_filter_agents,
                    )
                )
            
            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=agent_sensors,
            )
            
            # Initialize reward tracking
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            agent.is_active = torch.ones(batch_dim, dtype=torch.bool, device=device)
            
            world.add_agent(agent)
            
            # Create corresponding goal
            goal = Landmark(
                name=f"goal_{i}",
                collide=False,
                color=color,
                shape=Sphere(radius=self.agent_radius)
            )
            world.add_landmark(goal)
            agent.goal = goal
        
        # Initialize boundary walls
        self.init_boundary(world)
        
        # Initialize global reward tracking
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        
        return world
    
    def set_inactive_agents_to_origin(self, num_active_agents: int, env_index: int = None):
        """Set inactive agents to origin (0,0) positions"""
        self.current_active_agents = min(num_active_agents, self.n_agents)
        
        # Set inactive agents to origin
        for i in range(self.current_active_agents, self.n_agents):
            agent = self.world.agents[i]
            origin_pos = torch.tensor([0.0, 0.0], device=self.world.device)
            
            if env_index is None:
                agent.is_active[:] = False
                agent.set_pos(origin_pos, batch_index=env_index)
                agent.goal.set_pos(origin_pos, batch_index=env_index)
            else:
                agent.is_active[env_index] = False
                agent.state.pos[env_index] = origin_pos
                agent.goal.state.pos[env_index] = origin_pos
        
        # Ensure active agents are marked as active
        for i in range(self.current_active_agents):
            agent = self.world.agents[i]
            if env_index is None:
                agent.is_active[:] = True
            else:
                agent.is_active[env_index] = True
    
    def reset_world_at(self, env_index: int = 0, num_active_agents: int = None):
        """Enhanced reset with agent count specification and state persistence"""
        
        # Set number of active agents if specified
        if num_active_agents is not None:
            self.set_inactive_agents_to_origin(num_active_agents, env_index)
        
        if self.episode_count == 0:
            # First episode: initialize positions based on configuration
            if self.use_test_positions and self.test_start_positions is not None and self.test_goal_positions is not None:
                self._set_test_positions(env_index)
            else:
                self._initialize_first_episode(env_index)
        else:
            # Subsequent episodes: maintain positions for incomplete agents
            self._reset_with_state_persistence(env_index)
        
        # Reset walls
        self.reset_walls(env_index)
        self.episode_count += 1
    
    def _set_test_positions(self, env_index: int = None):
        """Set agent and goal positions from test case data"""
        # Convert to tensors if needed
        if isinstance(self.test_start_positions, (list, np.ndarray)):
            start_positions = torch.tensor(
                self.test_start_positions, dtype=torch.float32, device=self.world.device
            )
        else:
            start_positions = self.test_start_positions.to(self.world.device)
        
        if isinstance(self.test_goal_positions, (list, np.ndarray)):
            goal_positions = torch.tensor(
                self.test_goal_positions, dtype=torch.float32, device=self.world.device
            )
        else:
            goal_positions = self.test_goal_positions.to(self.world.device)
        
        # Set positions for active agents only
        for i in range(self.current_active_agents):
            agent = self.world.agents[i]
            
            # Set agent position
            agent.set_pos(start_positions[i], batch_index=env_index)
            
            # Set corresponding goal position
            agent.goal.set_pos(goal_positions[i], batch_index=env_index)
            
            # Initialize position shaping for reward calculation
            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos, dim=1
                    ) * self.pos_shaping_factor
                )
                # Store initial states
                self.agent_states_history[i] = agent.state.pos.clone()
                self.goal_states_history[i] = agent.goal.state.pos.clone()
                self.agent_reached_goal[i] = torch.zeros(agent.state.pos.shape[0], dtype=torch.bool, device=self.world.device)
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    ) * self.pos_shaping_factor
                )
                # Store initial states for specific environment
                if i not in self.agent_states_history:
                    self.agent_states_history[i] = torch.zeros_like(agent.state.pos)
                    self.goal_states_history[i] = torch.zeros_like(agent.goal.state.pos)
                    self.agent_reached_goal[i] = torch.zeros(agent.state.pos.shape[0], dtype=torch.bool, device=self.world.device)
                
                self.agent_states_history[i][env_index] = agent.state.pos[env_index].clone()
                self.goal_states_history[i][env_index] = agent.goal.state.pos[env_index].clone()
                self.agent_reached_goal[i][env_index] = False
    
    def _initialize_first_episode(self, env_index: int = None):
        """Initialize the first episode with random positions"""
        active_agents = [self.world.agents[i] for i in range(self.current_active_agents)]
        
        # Spawn active agents randomly
        ScenarioUtils.spawn_entities_randomly(
            active_agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (0.0 * self.world_semidim, 1 * self.world_semidim),
            (0.0 * self.world_semidim, 1 * self.world_semidim),
        )
        
        # Store initial positions
        for i, agent in enumerate(active_agents):
            if env_index is None:
                self.agent_states_history[i] = agent.state.pos.clone()
                self.agent_reached_goal[i] = torch.zeros(agent.state.pos.shape[0], dtype=torch.bool, device=self.world.device)
            else:
                if i not in self.agent_states_history:
                    self.agent_states_history[i] = torch.zeros_like(agent.state.pos)
                    self.agent_reached_goal[i] = torch.zeros(agent.state.pos.shape[0], dtype=torch.bool, device=self.world.device)
                self.agent_states_history[i][env_index] = agent.state.pos[env_index].clone()
                self.agent_reached_goal[i][env_index] = False
        
        # Set goals randomly
        occupied_positions = torch.stack([agent.state.pos for agent in active_agents], dim=1)
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)
        
        goal_poses = []
        for _ in active_agents:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-1.8 * self.world_semidim, -0.0 * self.world_semidim),
                y_bounds=(-1.8 * self.world_semidim, -0.0 * self.world_semidim),
            )
            goal_poses.append(position.squeeze(1))
            occupied_positions = torch.cat([occupied_positions, position], dim=1)
        
        # Assign goals and initialize shaping
        for i, agent in enumerate(active_agents):
            goal_index = self._get_goal_index(i)
            agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)
            
            # Store goal positions
            if env_index is None:
                self.goal_states_history[i] = agent.goal.state.pos.clone()
                agent.pos_shaping = (
                    torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=1)
                    * self.pos_shaping_factor
                )
            else:
                if i not in self.goal_states_history:
                    self.goal_states_history[i] = torch.zeros_like(agent.goal.state.pos)
                self.goal_states_history[i][env_index] = agent.goal.state.pos[env_index].clone()
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    ) * self.pos_shaping_factor
                )
    
    def _reset_with_state_persistence(self, env_index: int = None):
        """Reset while maintaining agent positions and incomplete goals"""
        for i in range(self.current_active_agents):
            agent = self.world.agents[i]
            
            # Check if agent reached goal in previous episode
            if i in self.agent_states_history and i in self.goal_states_history:
                if env_index is None:
                    # Check for all environments
                    agent_at_goal = torch.all(
                        torch.linalg.vector_norm(
                            self.agent_states_history[i] - self.goal_states_history[i], dim=1
                        ) < agent.goal.shape.radius
                    )
                    
                    if not agent_at_goal.any():
                        # Keep current position and goal
                        agent.set_pos(self.agent_states_history[i], batch_index=env_index)
                        agent.goal.set_pos(self.goal_states_history[i], batch_index=env_index)
                    else:
                        # Agent reached goal: assign new random goal
                        self._initialize_agent_randomly(agent, i, env_index)
                else:
                    # Check for specific environment
                    agent_at_goal = (
                        torch.linalg.vector_norm(
                            self.agent_states_history[i][env_index] - self.goal_states_history[i][env_index]
                        ) < agent.goal.shape.radius
                    )
                    
                    if not agent_at_goal:
                        # Keep current position and goal
                        agent.state.pos[env_index] = self.agent_states_history[i][env_index]
                        agent.goal.state.pos[env_index] = self.goal_states_history[i][env_index]
                    else:
                        # Agent reached goal: initialize randomly
                        self._initialize_agent_randomly(agent, i, env_index)
            else:
                # No history for this agent: initialize randomly
                self._initialize_agent_randomly(agent, i, env_index)
            
            # Update position shaping
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
    
    def _initialize_agent_randomly(self, agent: Agent, agent_idx: int, env_index: int = None):
        """Initialize a single agent randomly"""
        # Random agent position
        agent_pos = ScenarioUtils.find_random_pos_for_entity(
            occupied_positions=torch.zeros(1, 1, 2, device=self.world.device),
            env_index=env_index,
            world=self.world,
            min_dist_between_entities=self.min_distance_between_entities,
            x_bounds=(0.0 * self.world_semidim, 1 * self.world_semidim),
            y_bounds=(0.0 * self.world_semidim, 1 * self.world_semidim),
        )
        agent.set_pos(agent_pos.squeeze(1), batch_index=env_index)
        
        # Random goal position
        goal_pos = ScenarioUtils.find_random_pos_for_entity(
            occupied_positions=agent_pos,
            env_index=env_index,
            world=self.world,
            min_dist_between_entities=self.min_distance_between_entities,
            x_bounds=(-1.8 * self.world_semidim, -0.0 * self.world_semidim),
            y_bounds=(-1.8 * self.world_semidim, -0.0 * self.world_semidim),
        )
        agent.goal.set_pos(goal_pos.squeeze(1), batch_index=env_index)
    
    def _get_goal_index(self, agent_idx: int) -> int:
        """Get goal index based on goal sharing configuration"""
        if self.split_goals:
            return int(agent_idx // self.agents_with_same_goal)
        else:
            return 0 if agent_idx < self.agents_with_same_goal else agent_idx
    
    def init_boundary(self, world):
        """Initialize boundary walls"""
        top_wall = Landmark(
            name="Top Wall",
            collide=True,
            movable=False,
            shape=Line(length=self.floor_width),
            color=Color.BLACK,
        )
        world.add_landmark(top_wall)
        
        left_wall = Landmark(
            name="Left Wall",
            collide=True,
            movable=False,
            shape=Line(length=self.floor_width),
            color=Color.BLACK,
        )
        world.add_landmark(left_wall)
        
        bottom_wall = Landmark(
            name="Bottom Wall",
            collide=True,
            movable=False,
            shape=Line(length=self.floor_width),
            color=Color.BLACK,
        )
        world.add_landmark(bottom_wall)
        
        right_wall = Landmark(
            name="Right Wall",
            collide=True,
            movable=False,
            shape=Line(length=self.floor_width),
            color=Color.BLACK,
        )
        world.add_landmark(right_wall)
    
    def reset_walls(self, env_index: int = None):
        """Reset wall positions"""
        for landmark in self.world.landmarks:
            if landmark.name == "Top Wall":
                landmark.set_pos(
                    torch.tensor([0, self.floor_width / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([0], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Left Wall":
                landmark.set_pos(
                    torch.tensor([-self.floor_length, -0 * self.floor_width / 4], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Bottom Wall":
                landmark.set_pos(
                    torch.tensor([0, -self.floor_width / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([0], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Wall":
                landmark.set_pos(
                    torch.tensor([self.floor_length, -0 * self.floor_width / 4], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
    
    def observation(self, agent: Agent):
        """Modified observation to handle inactive agents"""
        agent_idx = self.world.agents.index(agent)
        
        # Return zero observation for inactive agents
        if agent_idx >= self.current_active_agents:
            obs_dim = self._get_observation_dim()
            return torch.zeros(agent.state.pos.shape[0], obs_dim, device=self.world.device)
        
        # Standard observation for active agents
        goal_poses = []
        if self.observe_all_goals:
            for i in range(self.current_active_agents):
                a = self.world.agents[i]
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:
            goal_poses.append(agent.state.pos - agent.goal.state.pos)
        
        obs_components = [agent.state.pos, agent.state.vel] + goal_poses
        
        if self.collisions and len(agent.sensors) > 0:
            obs_components.append(agent.sensors[0]._max_range - agent.sensors[0].measure())
        
        return torch.cat(obs_components, dim=-1)
    
    def _get_observation_dim(self) -> int:
        """Calculate observation dimension"""
        base_dim = 4  # pos + vel
        if self.observe_all_goals:
            goal_dim = 2 * self.current_active_agents
        else:
            goal_dim = 2
        lidar_dim = 20 if self.collisions else 0
        return base_dim + goal_dim + lidar_dim
    
    def reward(self, agent: Agent):
        """Modified reward calculation for active agents only"""
        agent_idx = self.world.agents.index(agent)
        if agent_idx >= self.current_active_agents:
            return torch.zeros_like(getattr(agent, 'pos_rew', torch.zeros(1, device=self.world.device)))
        
        is_first = agent == self.world.agents[0]
        
        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0
            
            # Only consider active agents for rewards
            active_agents = [self.world.agents[i] for i in range(self.current_active_agents)]
            
            for a in active_agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0
            
            # Check if all active agents reached goals
            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in active_agents], dim=-1),
                dim=-1,
            )
            
            self.final_rew[self.all_goal_reached] = self.final_reward
            
            # Check collisions only among active agents
            for i, a in enumerate(active_agents):
                for j, b in enumerate(active_agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty
                        b.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty
        
        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew
    
    def agent_reward(self, agent: Agent):
        """Calculate individual agent reward"""
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius
        
        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        
        return agent.pos_rew
    
    def done(self):
        """Check if episode is done (all active agents reached goals)"""
        active_agents = [self.world.agents[i] for i in range(self.current_active_agents)]
        return torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                ) < agent.shape.radius
                for agent in active_agents
            ],
            dim=-1,
        ).all(-1)
    
    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """Return information dictionary"""
        agent_idx = self.world.agents.index(agent)
        if agent_idx >= self.current_active_agents:
            # Return zero info for inactive agents
            zero_tensor = torch.zeros_like(getattr(agent, 'pos_rew', torch.zeros(1, device=self.world.device)))
            return {
                "pos_rew": zero_tensor,
                "final_rew": zero_tensor,
                "agent_collisions": zero_tensor,
            }
        
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
        }
    
    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        """Additional rendering for active agents only"""
        from vmas.simulator import rendering
        
        geoms: List[Geom] = []
        
        # Communication lines between active agents only
        active_agents = [self.world.agents[i] for i in range(self.current_active_agents)]
        
        for i, agent1 in enumerate(active_agents):
            for j, agent2 in enumerate(active_agents):
                if j <= i:
                    continue
                
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)
        
        return geoms


# Set the main scenario class for compatibility
Scenario = DynamicNavigationScenario

if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
