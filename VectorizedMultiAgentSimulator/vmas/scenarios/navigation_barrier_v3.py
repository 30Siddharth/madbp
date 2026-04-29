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

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 6)
        self.n_obs = kwargs.pop("n_obs", 0)
        self.collisions = kwargs.pop("collisions", True)
        self.agents_with_same_goal = kwargs.pop("agents_with_same_goal", 1)
        self.split_goals = kwargs.pop("split_goals", False)
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)
        self.lidar_range = kwargs.pop("lidar_range", 0.15)
        self.agent_radius = kwargs.pop("agent_radius", 0.05)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.shared_rew = kwargs.pop("shared_rew", False)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 0.5)
        self.final_reward = kwargs.pop("final_reward", 1.0)
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 1

        self.min_collision_distance = 0.05
        self.viewer_size = (800, 600)
        self.viewer_zoom = 1.5
        self.render_origin = (0.0, 0.0)

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
                shape=Sphere(radius=self.agent_radius),
            )
            world.add_landmark(goal)
            agent.goal = goal

        # Add central barrier
        # self.central_barrier = Landmark(
        #     name="central_barrier",
        #     collide=True,
        #     movable=False,
        #     shape=Line(length=self.floor_width * 0.2),  # 60% of environment width
        #     color=Color.BLACK,
        # )
        # world.add_landmark(self.central_barrier)

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

        center_barrier_horizontal = Landmark(
            name="Central Barrier H",
            collide=True,
            movable=False,
            shape=Line(length=self.floor_width * 0.2),
            color=Color.BLACK,
        )
        world.add_landmark(center_barrier_horizontal)

        center_barrier_vertical = Landmark(
            name="Central Barrier V",
            collide=True,
            movable=False,
            shape=Line(length=self.floor_width * 0.2),
            color=Color.BLACK,
        )
        world.add_landmark(center_barrier_vertical)

    def reset_world_at(self, env_index: int = None):
        """Position agents and goals in different sectors with consistent random distribution"""
        device = self.world.device
        batch_dim = self.world.batch_dim
        
        circle_radius = self.floor_width * 0.3
        center = torch.tensor([0.0, 0.0], device=device)
        
        # Create sector divisions - minimum 4 sectors, or more if needed
        n_sectors = max(4, self.n_agents)
        sector_size = 2 * torch.pi / n_sectors
        
        agent_positions = []
        goal_positions = []
        
        # Generate random offsets for this episode (consistent across batch)
        agent_offsets = torch.rand(self.n_agents, device=device) * 0.6 + 0.2  # Between 0.2 and 0.8
        goal_offsets = torch.rand(self.n_agents, device=device) * 0.6 + 0.2   # Between 0.2 and 0.8
    
        for i, agent in enumerate(self.world.agents):
            # Agent positioned in sector i with random offset
            agent_sector = i
            agent_angle = torch.tensor(
                agent_sector * sector_size + sector_size * agent_offsets[i], 
                device=device
            )
            
            # Goal positioned in a different sector (+2 sectors away, wrapping around)
            goal_sector = (agent_sector + 2) % n_sectors
            goal_angle = torch.tensor(
                goal_sector * sector_size + sector_size * goal_offsets[i], 
                device=device
            )
            
            # Calculate positions on circle
            agent_offset = circle_radius * torch.stack([torch.cos(agent_angle), torch.sin(agent_angle)])
            agent_pos = center + agent_offset
            
            goal_offset = circle_radius * torch.stack([torch.cos(goal_angle), torch.sin(goal_angle)])
            goal_pos = center + goal_offset
            
            # Expand for batch dimension
            if env_index is None:
                agent_pos_batch = agent_pos.unsqueeze(0).expand(batch_dim, -1)
                goal_pos_batch = goal_pos.unsqueeze(0).expand(batch_dim, -1)
            else:
                agent_pos_batch = agent_pos.unsqueeze(0)
                goal_pos_batch = goal_pos.unsqueeze(0)
            
            agent_positions.append(agent_pos_batch)
            goal_positions.append(goal_pos_batch)
            
            # Set agent position
            agent.set_pos(agent_pos_batch, batch_index=env_index)
    
        # Set goal positions based on sharing configuration
        for i, agent in enumerate(self.world.agents):
            if self.split_goals:
                goal_index = int(i // self.agents_with_same_goal)
            else:
                goal_index = 0 if i < self.agents_with_same_goal else i
            
            agent.goal.set_pos(goal_positions[goal_index], batch_index=env_index)
            
            # Update position shaping
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

        
        self.reset_walls(env_index)

    def reset_walls(self, env_index: int = None):
        for landmark in self.world.landmarks:
            if landmark.name == "Central Barrier H":
                # Position barrier at center horizontally
                landmark.set_pos(
                    torch.tensor(
                        [0.0, 0.0],  # Center of environment
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor(
                        [0],  # Horizontal orientation
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            elif landmark.name == "Central Barrier V":
                # Position barrier at center horizontally
                landmark.set_pos(
                    torch.tensor(
                        [0.0, 0.0],  # Center of environment
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor(
                        [torch.pi / 2],  # Horizontal orientation
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            elif landmark.name == "Top Wall":
                landmark.set_pos(
                    torch.tensor(
                        [0, self.floor_width / 2],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([0], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Left Wall":
                landmark.set_pos(
                    torch.tensor(
                        [-self.floor_length / 1, 0],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Bottom Wall":
                landmark.set_pos(
                    torch.tensor(
                        [0, -self.floor_width / 2],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([0], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Wall":
                landmark.set_pos(
                    torch.tensor(
                        [self.floor_length / 1, 0],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
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
                a.agent_collision_rew[:] = -0.01

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


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, clf_epsilon=0.8, clf_slack=50.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf_epsilon = clf_epsilon
        self.clf_slack = clf_slack

    def compute_action(self, observation: Tensor, u_range: Tensor) -> Tensor:
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer

        self.n_env = observation.shape[0]
        self.device = observation.device

        agent_pos = observation[:, :2]
        agent_vel = observation[:, 2:4]
        goal_pos = (1.0) * (observation[:, 4:6] - agent_pos)

        # Lyapunov Function
        V_value = (
            (agent_pos[:, X] - goal_pos[:, X]) ** 2
            + 0.5 * (agent_pos[:, X] - goal_pos[:, X]) * agent_vel[:, X]
            + agent_vel[:, X] ** 2
            + (agent_pos[:, Y] - goal_pos[:, Y]) ** 2
            + 0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) * agent_vel[:, Y]
            + agent_vel[:, Y] ** 2
        )

        LfV_val = (2 * (agent_pos[:, X] - goal_pos[:, X]) + agent_vel[:, X]) * (
            agent_vel[:, X]
        ) + (2 * (agent_pos[:, Y] - goal_pos[:, Y]) + agent_vel[:, Y]) * (
            agent_vel[:, Y]
        )

        LgV_vals = torch.stack(
            [
                0.5 * (agent_pos[:, X] - goal_pos[:, X]) + 2 * agent_vel[:, X],
                0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) + 2 * agent_vel[:, Y],
            ],
            dim=1,
        )

        # Define QP
        u = cp.Variable(2)
        V_param = cp.Parameter(1)
        lfV_param = cp.Parameter(1)
        lgV_params = cp.Parameter(2)
        clf_slack = cp.Variable(1)

        constraints = []
        qp_objective = cp.Minimize(cp.sum_squares(u) + self.clf_slack * clf_slack**2)

        constraints += [u <= u_range]
        constraints += [u >= -u_range]
        constraints += [
            lfV_param + lgV_params @ u + self.clf_epsilon * V_param + clf_slack <= 0
        ]

        QP_problem = cp.Problem(qp_objective, constraints)
        QP_controller = CvxpyLayer(
            QP_problem,
            parameters=[V_param, lfV_param, lgV_params],
            variables=[u],
        )

        CVXpylayer_parameters = [
            V_value.unsqueeze(1),
            LfV_val.unsqueeze(1),
            LgV_vals,
        ]

        action = QP_controller(*CVXpylayer_parameters, solver_args={"max_iters": 500})[0]
        action = torch.clamp(action, -u_range, u_range)

        return action


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
