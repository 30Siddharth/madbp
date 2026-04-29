# from navigation_v3 import HeuristicPolicy
from vmas.scenarios.navigation_v3 import HeuristicPolicy as NavigationHeuristic
import torch
import numpy as np

class CLFQPController:
    """
    Enhanced CLF-QP controller using the HeuristicPolicy from navigation_v3
    """
    
    def __init__(self, n_agents: int, device: str = 'cuda', clf_epsilon: float = 0.9, clf_slack: float = 1.0):
        self.n_agents = n_agents
        self.device = device
        self.goal_tolerance = 0.05
        
        # Initialize the HeuristicPolicy for each agent
        self.heuristic_policies = []
        for i in range(n_agents):
            policy = NavigationHeuristic(
                clf_epsilon=clf_epsilon,
                clf_slack=clf_slack,
                continuous_action=True
            )
            self.heuristic_policies.append(policy)
    
    def compute_action(self, current_pos: torch.Tensor, current_vel: torch.Tensor, 
                      goal_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute control actions using the navigation HeuristicPolicy
        
        Args:
            current_pos: Current positions [n_agents, 2]
            current_vel: Current velocities [n_agents, 2]  
            goal_pos: Goal positions [n_agents, 2]
            
        Returns:
            Control forces [n_agents, 2]
        """
        batch_size = current_pos.shape[0] if len(current_pos.shape) > 2 else 1
        if len(current_pos.shape) == 2:
            current_pos = current_pos.unsqueeze(0)
            current_vel = current_vel.unsqueeze(0)
            goal_pos = goal_pos.unsqueeze(0)
        
        actions = torch.zeros(batch_size, self.n_agents, 2, device=self.device)
        
        for batch_idx in range(batch_size):
            for agent_idx in range(self.n_agents):
                # Construct observation vector for HeuristicPolicy
                # Format: [pos_x, pos_y, vel_x, vel_y, goal_x, goal_y]
                observation = torch.cat([
                    current_pos[batch_idx, agent_idx],  # agent position
                    current_vel[batch_idx, agent_idx],  # agent velocity
                    goal_pos[batch_idx, agent_idx]      # goal position
                ]).unsqueeze(0)  # Add batch dimension
                
                # Control range (from VMAS navigation scenario)
                u_range = torch.tensor([2.0, 2.0], device=self.device)
                
                # Compute action using HeuristicPolicy
                try:
                    action = self.heuristic_policies[agent_idx].compute_action(
                        observation, u_range
                    )
                    actions[batch_idx, agent_idx] = action.squeeze(0)
                except Exception as e:
                    # Fallback to simple PD control if QP fails
                    print(f"QP solver failed for agent {agent_idx}: {e}")
                    pos_error = goal_pos[batch_idx, agent_idx] - current_pos[batch_idx, agent_idx]
                    vel_error = -current_vel[batch_idx, agent_idx]
                    actions[batch_idx, agent_idx] = 25.0 * pos_error + 3.0 * vel_error
                    actions[batch_idx, agent_idx] = torch.clamp(actions[batch_idx, agent_idx], -2.0, 2.0)
        
        return actions.squeeze(0) if batch_size == 1 else actions
