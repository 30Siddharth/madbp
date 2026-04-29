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

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Configuration parameters
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 2)
        self.collisions = kwargs.pop("collisions", True)
        self.agents_with_same_goal = kwargs.pop("agents_with_same_goal", 1)
        self.split_goals = kwargs.pop("split_goals", False)
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)
        self.lidar_range = kwargs.pop("lidar_range", 0.15)
        self.agent_radius = kwargs.pop("agent_radius", 0.05)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.shared_rew = kwargs.pop("shared_rew", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 1.0)
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.1)

        # Environment dimensions
        self.floor_width = kwargs.pop("width", 2.0)
        self.floor_length = kwargs.pop("length", 4.0)

        # Corridor configuration
        self.n_corridors = 3
        self.corridor_width = self.floor_width / self.n_corridors
        self.wall_thickness = 0.1

        # Test position parameters
        self.test_start_positions = kwargs.pop("test_start_positions", None)
        self.test_goal_positions = kwargs.pop("test_goal_positions", None)
        self.use_test_positions = kwargs.pop("use_test_positions", False)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.1
        self.viewer_size = (800, 600)
        self.viewer_zoom = 2.0
        self.render_origin = (0.0, 0.0)

        # Create world
        world = World(batch_dim, device, substeps=2)

        # Color scheme
        known_colors = [color.value for color in Color]

        # Entity filter for lidar
        entity_filter_agents: Callable[[Entity], bool] = lambda e: (
            isinstance(e, Agent) or
            (isinstance(e, Landmark) and getattr(e, "collide", False))
        )

        # Add agents
        for i in range(self.n_agents):
            color = known_colors[i % len(known_colors)]

            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                u_range=2.0,
                sensors=(
                    Lidar(
                        world,
                        n_rays=20,
                        max_range=self.lidar_range,
                        entity_filter=entity_filter_agents,
                    ),
                ) if self.collisions else None,
            )

            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

        # Add goals
        for i in range(self.n_agents):
            color = known_colors[i % len(known_colors)]
            goal = Landmark(
                name=f"goal_{i}",
                collide=False,
                color=color,
                shape=Sphere(radius=self.agent_radius),
            )
            world.add_landmark(goal)
            world.agents[i].goal = goal

        # Create boundary walls and corridor dividers
        self.init_boundary(world)
        self.create_corridor_walls(world)

        # Initialize reward tracking
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def init_boundary(self, world):
        """Create boundary walls using Line landmarks"""
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
            shape=Line(length=self.floor_length),
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
            shape=Line(length=self.floor_length),
            color=Color.BLACK,
        )
        world.add_landmark(right_wall)

    def create_corridor_walls(self, world):
        """Create vertical corridor dividing walls using Box landmarks"""
        # First corridor divider (between corridor 0 and 1)
        corridor_wall1 = Landmark(
            name="Corridor Wall 1",
            collide=True,
            movable=False,
            shape=Box(length=0.75*self.floor_length, width=self.wall_thickness),
            color=Color.RED,
        )
        world.add_landmark(corridor_wall1)

        # Second corridor divider (between corridor 1 and 2)
        corridor_wall2 = Landmark(
            name="Corridor Wall 2",
            collide=True,
            movable=False,
            shape=Box(length=0.75*self.floor_length, width=self.wall_thickness),
            color=Color.RED,
        )
        world.add_landmark(corridor_wall2)

    def reset_world_at(self, env_index: int = None):
        """Reset agent and goal positions for corridor navigation"""
        if self.use_test_positions and self.test_start_positions is not None and self.test_goal_positions is not None:
            # Use fixed test positions
            self._set_test_positions(env_index)
        else:
            # Use corridor-based positioning (original behavior)
            self._set_corridor_positions(env_index)

        # Position boundary walls
        self.reset_boundary_walls(env_index)
        # Position corridor dividing walls
        self.reset_corridor_walls(env_index)

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

        # Set agent positions
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                start_positions[i],
                batch_index=env_index
            )

        # Set goal positions
        for i, agent in enumerate(self.world.agents):
            agent.goal.set_pos(
                goal_positions[i],
                batch_index=env_index
            )

        # Initialize position shaping for reward calculation
        for agent in self.world.agents:
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

    def _set_corridor_positions(self, env_index: int = None):
        """Original corridor positioning logic"""
        # Position agents at the start of corridors (bottom end)
        for i, agent in enumerate(self.world.agents):
            corridor_index = i % self.n_corridors
            x_pos = (-self.floor_width/2 + self.corridor_width/2 +
                    corridor_index * self.corridor_width)
            y_pos = -self.floor_length/2 + 0.3  # Start position with buffer

            agent.set_pos(
                torch.tensor([x_pos, y_pos], dtype=torch.float32, device=self.world.device),
                batch_index=env_index
            )

        # Position goals at the end of corridors (top end)
        for i, agent in enumerate(self.world.agents):
            corridor_index = i % self.n_corridors
            x_pos = (-self.floor_width/2 + self.corridor_width/2 +
                    corridor_index * self.corridor_width)
            y_pos = self.floor_length/2 - 0.3  # Goal position with buffer

            agent.goal.set_pos(
                torch.tensor([x_pos, y_pos], dtype=torch.float32, device=self.world.device),
                batch_index=env_index
            )

        # Initialize position shaping
        for agent in self.world.agents:
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

    def reset_boundary_walls(self, env_index: int = None):
        """Position boundary walls"""
        for landmark in self.world.landmarks:
            if landmark.name == "Top Wall":
                landmark.set_pos(
                    torch.tensor([0, self.floor_length/2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index
                )
                landmark.set_rot(
                    torch.tensor([0], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index
                )
            elif landmark.name == "Left Wall":
                landmark.set_pos(
                    torch.tensor([-self.floor_width/2, 0], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index
                )
                landmark.set_rot(
                    torch.tensor([torch.pi/2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index
                )
            elif landmark.name == "Bottom Wall":
                landmark.set_pos(
                    torch.tensor([0, -self.floor_length/2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index
                )
                landmark.set_rot(
                    torch.tensor([0], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index
                )
            elif landmark.name == "Right Wall":
                landmark.set_pos(
                    torch.tensor([self.floor_width/2, 0], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index
                )
                landmark.set_rot(
                    torch.tensor([torch.pi/2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index
                )

    def reset_corridor_walls(self, env_index: int = None):
        """Position corridor dividing walls to create vertical corridors"""
        for landmark in self.world.landmarks:
            if landmark.name == "Corridor Wall 1":
                # Position between corridor 0 and 1
                x_pos = -self.floor_width/2 + self.corridor_width
                landmark.set_pos(
                    torch.tensor([x_pos, 0], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Corridor Wall 2":
                # Position between corridor 1 and 2
                x_pos = -self.floor_width/2 + 2 * self.corridor_width
                landmark.set_pos(
                    torch.tensor([x_pos, 0], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )

    def reward(self, agent: Agent):
        """Compute rewards for corridor navigation"""
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0

            # Check if all agents reached their goals
            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )
            self.final_rew[self.all_goal_reached] = self.final_reward

            # Agent collision penalties
            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        collision_mask = distance <= self.min_collision_distance
                        a.agent_collision_rew[collision_mask] += self.agent_collision_penalty
                        b.agent_collision_rew[collision_mask] += self.agent_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    def agent_reward(self, agent: Agent):
        """Calculate individual agent reward based on progress toward goal"""
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos, dim=-1
        )

        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping

        return agent.pos_rew

    def observation(self, agent: Agent):
        """Generate agent observation including position, velocity, goal, and lidar"""
        goal_poses = []
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:
            goal_poses.append(agent.goal.state.pos)

        observations = [
            agent.state.pos,
            agent.state.vel,
        ] + goal_poses

        if self.collisions and agent.sensors:
            lidar_data = agent.sensors[0]._max_range - agent.sensors[0].measure()
            observations.append(lidar_data)

        return torch.cat(observations, dim=-1)

    def done(self):
        """Check if all agents have reached their goals"""
        return torch.stack([
            torch.linalg.vector_norm(
                agent.state.pos - agent.goal.state.pos, dim=-1
            ) < agent.shape.radius
            for agent in self.world.agents
        ], dim=-1).all(-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """Return additional information for logging and analysis"""
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        """Additional rendering for communication lines"""
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

if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
