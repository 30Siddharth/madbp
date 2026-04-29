# Test Scanrio for navigation.
# Owner: Siddharth Singh

import typing
from typing import Callable, Dict, List
import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World, Line, Box
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y
import numpy as np

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 4)
        self.n_obs = kwargs.pop("n_obs", 0)
        self.collisions = kwargs.pop("collisions", True)
        self.agents_with_same_goal = kwargs.pop("agents_with_same_goal", 1)
        self.split_goals = kwargs.pop("split_goals", False)
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)
        
        self.lidar_range = kwargs.pop("lidar_range", 0.15)
        self.agent_radius = kwargs.pop("agent_radius", 0.1)
        self.comms_range = kwargs.pop("comms_range", 0)

        self.shared_rew = kwargs.pop("shared_rew", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 0.01)
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)

        self.test_start_positions = kwargs.pop("test_start_positions", None)
        self.test_goal_positions = kwargs.pop("test_goal_positions", None)
        self.use_test_positions = kwargs.pop("use_test_positions", False)

        self.viewer_size = (800, 600)  # Set based on your window limits
        self.viewer_zoom = 1.5  # Adjust zoom level for optimal view
        self.render_origin = (0.0, 0.0)  # Fixed center point for camera

        ScenarioUtils.check_kwargs_consumed(kwargs)
        
        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 1
        self.min_collision_distance = 0.05
        self.floor_width = kwargs.pop("width", 4)
        self.floor_length = kwargs.pop("length", 2)
        
        assert 1 <= self.agents_with_same_goal <= self.n_agents
        
        if self.agents_with_same_goal > 1:
            assert (
                not self.collisions
            ), "If agents share goals they cannot be collidables"
            
        if self.split_goals:
            assert (
                self.n_agents % 2 == 0
                and self.agents_with_same_goal == self.n_agents // 2
            ), "Splitting the goals is allowed when the agents are even and half the team has the same goal"
            
        # Make world
        world = World(batch_dim, device, substeps=2)
        known_colors = [color.value for color in Color]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )
        
        entity_filter_agents: Callable[[Entity], bool] = lambda e: (
            isinstance(e, Agent) or (isinstance(e, Landmark) and getattr(e, "collide", False))
        )
        
        # Add agents
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )
            
            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=(
                    Lidar(
                        world,
                        n_rays=20,
                        max_range=self.lidar_range,
                        entity_filter=entity_filter_agents,
                    ),
                )
                if self.collisions
                else None,
            )
            
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)
            
            # Add goals
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=color,
            )
            world.add_landmark(goal)
            agent.goal = goal
            
        # Make the bounding walls
        self.init_boundary(world)
        
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        
        return world
        
    def init_boundary(self, world):
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
        
    def reset_world_at(self, env_index: int = 0):
        if self.use_test_positions and self.test_start_positions is not None and self.test_goal_positions is not None:
            # Use fixed test positions
            self._set_test_positions(env_index)
        else:
            # Use existing random positioning (your original code)
            self._set_random_positions(env_index)
        
        self.reset_walls(env_index)

    def _set_test_positions(self, env_index: int = None):
        """Set agent and goal positions from test case data"""
        # Convert to tensors if needed
        if isinstance(self.test_start_positions, (list, np.ndarray)):
            start_positions = torch.tensor(self.test_start_positions, dtype=torch.float32, device=self.world.device)
        else:
            start_positions = self.test_start_positions.to(self.world.device)
        
        if isinstance(self.test_goal_positions, (list, np.ndarray)):
            goal_positions = torch.tensor(self.test_goal_positions, dtype=torch.float32, device=self.world.device)
        else:
            goal_positions = self.test_goal_positions.to(self.world.device)
        
        # Set agent positions
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                start_positions[i],
                batch_index=env_index
            )
            
            # Set corresponding goal positions
            agent.goal.set_pos(
                goal_positions[i],
                batch_index=env_index
            )
            
            # Initialize position shaping for reward calculation
            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos, dim=1
                    ) * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    ) * self.pos_shaping_factor
                )

    def _set_random_positions(self, env_index: int = None):
        """Your original random positioning code"""
        # Spawn agents randomly
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (0.0*self.world_semidim, 2*self.world_semidim),
            (0.0*self.world_semidim, 2*self.world_semidim),
        )
        
        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)
            
        # Randomly Position Goals
        goal_poses = []
        for _ in self.world.agents:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-2*self.world_semidim, -1.0*self.world_semidim),
                y_bounds=(-2*self.world_semidim, -1.0*self.world_semidim),
            )
            goal_poses.append(position.squeeze(1))
            occupied_positions = torch.cat([occupied_positions, position], dim=1)
            
        # Reset Agents
        for i, agent in enumerate(self.world.agents):
            if self.split_goals:
                goal_index = int(i // self.agents_with_same_goal)
            else:
                goal_index = 0 if i < self.agents_with_same_goal else i
                
            agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)
            
            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos,
                        dim=1,
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )

        
    def reset_walls(self, env_index: int = None):
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
                    torch.tensor([-self.floor_length, -0*self.floor_width / 4], dtype=torch.float32, device=self.world.device),
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
                    torch.tensor([self.floor_length, -0*self.floor_width / 4], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
                
    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        
        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0
            
            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0
                
            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )
            
            self.final_rew[self.all_goal_reached] = self.final_reward
            
            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
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
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius
        
        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        
        return agent.pos_rew
        
    def observation(self, agent: Agent):
        goal_poses = []
        
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:
            goal_poses.append(agent.goal.state.pos)
            
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
            ]
            + goal_poses
            + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.collisions
                else []
            ),
            dim=-1,
        )
        
    def done(self):
        return torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                )
                < agent.shape.radius
                for agent in self.world.agents
            ],
            dim=-1,
        ).all(-1)
        
    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
        }
        
    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering
        
        geoms: List[Geom] = []
        
        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
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
