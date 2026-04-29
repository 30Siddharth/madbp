import asyncio
import threading
import queue
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class AgentEvent(Enum):
    ADD_AGENT = "add_agent"
    REMOVE_AGENT = "remove_agent"
    UPDATE_CONFIG = "update_config"
    HORIZON_COMPLETE = "horizon_complete"

@dataclass
class AgentCommand:
    event_type: AgentEvent
    agent_id: Optional[int] = None
    position: Optional[torch.Tensor] = None
    goal: Optional[torch.Tensor] = None
    timestamp: float = None

class DynamicVMASManager:
    """
    Core simulation manager that handles dynamic agent allocation
    and coordinates between service/client and pub/sub systems
    """
    
    def __init__(self, max_agents=10, initial_agents=4, horizon_size=8):
        self.max_agents = max_agents
        self.current_agents = initial_agents
        self.horizon_size = horizon_size
        self.active_agents = set(range(initial_agents))
        
        # Communication components
        self.command_queue = queue.Queue()
        self.state_publisher = StatePublisher()
        self.agent_service = AgentManagementService()
        self.horizon_service = HorizonManagementService()
        
        # VMAS components (adapted from your code)
        self.vmas_executor = None
        self.diffusion_model = None
        
        # Threading and async management
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        
    async def initialize_simulation(self, scenario_config: Dict):
        """Initialize the VMAS environment with dynamic capabilities"""
        
        # Initialize VMAS with maximum agent capacity
        self.vmas_executor = DynamicMovingHorizonExecutor(
            scenario_name=scenario_config.get('scenario', 'navigation_v3_test'),
            max_agents=self.max_agents,
            current_agents=self.current_agents,
            horizon_size=self.horizon_size,
            device=scenario_config.get('device', 'cuda')
        )
        
        # Load diffusion model
        await self._load_diffusion_model(scenario_config)
        
        # Start communication services
        await self._start_services()
        
    async def _load_diffusion_model(self, config: Dict):
        """Load the diffusion model for trajectory generation"""
        from MADP_diffusion_v2_2 import EnhancedMultiAgentDiffusionModel
        
        self.diffusion_model = EnhancedMultiAgentDiffusionModel(
            max_agents=self.max_agents,
            horizon=self.horizon_size,
            state_dim=2,
            img_ch=3,
            hid=128,
            diffusion_steps=config.get('diffusion_steps', 500),
            schedule_type='linear'
        ).to(config.get('device', 'cuda'))
        
        # Load pre-trained weights
        model_path = config.get('model_path', 'boundary_constrained_madp_navigation_v3.pth')
        self.diffusion_model.load_state_dict(torch.load(model_path))
        self.diffusion_model.eval()
        
    async def _start_services(self):
        """Start all communication services"""
        self.running = True
        
        # Start service/client handlers
        asyncio.create_task(self._agent_management_loop())
        asyncio.create_task(self._horizon_management_loop())
        
        # Start pub/sub handlers
        asyncio.create_task(self._state_publishing_loop())
        asyncio.create_task(self._command_processing_loop())

class AgentManagementService:
    """Service/Client pattern for critical agent operations"""
    
    def __init__(self):
        self.pending_requests = {}
        self.request_id = 0
        
    async def add_agent_request(self, position: torch.Tensor, goal: torch.Tensor) -> int:
        """Handle agent addition request with confirmation"""
        request_id = self.request_id
        self.request_id += 1
        
        # Create agent addition command
        command = AgentCommand(
            event_type=AgentEvent.ADD_AGENT,
            position=position,
            goal=goal,
            timestamp=time.time()
        )
        
        # Store pending request
        self.pending_requests[request_id] = {
            'command': command,
            'status': 'pending',
            'result': None
        }
        
        return request_id
        
    async def remove_agent_request(self, agent_id: int) -> int:
        """Handle agent removal request with confirmation"""
        request_id = self.request_id
        self.request_id += 1
        
        command = AgentCommand(
            event_type=AgentEvent.REMOVE_AGENT,
            agent_id=agent_id,
            timestamp=time.time()
        )
        
        self.pending_requests[request_id] = {
            'command': command,
            'status': 'pending',
            'result': None
        }
        
        return request_id
        
    async def get_request_status(self, request_id: int) -> Dict:
        """Get status of pending request"""
        return self.pending_requests.get(request_id, {'status': 'not_found'})

class StatePublisher:
    """Publisher/Subscriber pattern for real-time state distribution"""
    
    def __init__(self):
        self.subscribers = {}
        self.topics = {
            'agent_states': [],
            'environment_events': [],
            'trajectory_updates': [],
            'performance_metrics': []
        }
        
    def subscribe(self, topic: str, callback):
        """Subscribe to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        
    async def publish(self, topic: str, data: Dict):
        """Publish data to topic subscribers"""
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    await callback(data)
                except Exception as e:
                    print(f"Error in subscriber callback: {e}")

class DynamicMovingHorizonExecutor(MovingHorizonTrajectoryExecutor):
    """Extended executor with dynamic agent capabilities"""
    
    def __init__(self, scenario_name, max_agents, current_agents, horizon_size, device):
        # Initialize with maximum capacity
        super().__init__(
            scenario_name=scenario_name,
            num_envs=1,
            max_steps=200,  # Extended for long-running simulation
            n_agents=max_agents,  # Use max for VMAS initialization
            device=device,
            horizon_size=horizon_size
        )
        
        self.max_agents = max_agents
        self.current_agents = current_agents
        self.active_agent_mask = torch.zeros(max_agents, dtype=torch.bool, device=device)
        self.active_agent_mask[:current_agents] = True
        
        # Agent state tracking
        self.agent_positions = torch.zeros(max_agents, 2, device=device)
        self.agent_goals = torch.zeros(max_agents, 2, device=device)
        self.agent_spawn_times = {}
        
    async def add_agent_dynamic(self, position: torch.Tensor, goal: torch.Tensor) -> bool:
        """Dynamically add an agent during simulation"""
        
        # Find next available slot
        available_slots = (~self.active_agent_mask).nonzero(as_tuple=True)[0]
        if len(available_slots) == 0:
            return False  # No available slots
            
        agent_id = available_slots[0].item()
        
        # Activate agent
        self.active_agent_mask[agent_id] = True
        self.agent_positions[agent_id] = position
        self.agent_goals[agent_id] = goal
        self.agent_spawn_times[agent_id] = time.time()
        self.current_agents += 1
        
        # Update VMAS environment state
        await self._update_vmas_agent_state(agent_id, position, goal)
        
        return True
        
    async def remove_agent_dynamic(self, agent_id: int) -> bool:
        """Dynamically remove an agent during simulation"""
        
        if not self.active_agent_mask[agent_id]:
            return False  # Agent not active
            
        # Deactivate agent
        self.active_agent_mask[agent_id] = False
        self.current_agents -= 1
        
        # Clean up agent state
        if agent_id in self.agent_spawn_times:
            del self.agent_spawn_times[agent_id]
            
        # Update VMAS environment
        await self._update_vmas_agent_removal(agent_id)
        
        return True
        
    async def _update_vmas_agent_state(self, agent_id: int, position: torch.Tensor, goal: torch.Tensor):
        """Update VMAS environment with new agent state"""
        
        # Set agent position in VMAS
        if agent_id < len(self.env.agents):
            self.env.agents[agent_id].set_pos(position.unsqueeze(0))
            self.env.agents[agent_id].goal.set_pos(goal.unsqueeze(0))
            
    async def execute_dynamic_moving_horizon(self, model, frames, starts, goals, 
                                           total_duration: int = 1000):
        """Execute moving horizon with dynamic agent management"""
        
        print(f"Starting dynamic moving horizon execution for {total_duration} steps")
        
        # Initialize tracking
        execution_history = []
        agent_change_log = []
        performance_metrics = []
        
        current_step = 0
        horizon_count = 0
        
        while current_step < total_duration:
            
            # Check for scheduled agent changes
            await self._check_scheduled_changes(current_step, agent_change_log)
            
            # Generate current horizon
            active_agents = self.active_agent_mask.sum().item()
            
            if active_agents > 0:
                # Prepare inputs for active agents only
                active_starts = self.agent_positions[self.active_agent_mask].unsqueeze(0)
                active_goals = self.agent_goals[self.active_agent_mask].unsqueeze(0)
                
                # Generate trajectory segment
                horizon_start_time = time.time()
                
                with torch.no_grad():
                    horizon_prediction = model.sample_with_constraints(
                        frames, 
                        active_starts, 
                        active_goals, 
                        torch.tensor([active_agents]),
                        steps=50,
                        max_step_size=0.1
                    )
                
                generation_time = time.time() - horizon_start_time
                
                # Execute horizon segment
                horizon_results = await self._execute_horizon_segment(
                    horizon_prediction, active_agents, current_step
                )
                
                # Record metrics
                performance_metrics.append({
                    'step': current_step,
                    'active_agents': active_agents,
                    'generation_time': generation_time,
                    'tracking_error': horizon_results.get('tracking_error', 0),
                    'total_reward': horizon_results.get('total_reward', 0)
                })
                
                execution_history.append(horizon_results)
                horizon_count += 1
                
            current_step += self.horizon_size
            
            # Publish state updates
            await self._publish_state_update(current_step, active_agents, performance_metrics[-1] if performance_metrics else {})
            
        return {
            'execution_history': execution_history,
            'agent_changes': agent_change_log,
            'performance_metrics': performance_metrics,
            'total_horizons': horizon_count,
            'final_agents': active_agents
        }
        
    async def _check_scheduled_changes(self, current_step: int, change_log: List):
        """Check and execute scheduled agent additions/removals"""
        
        # Example scheduling logic - add agents at specific intervals
        if current_step % 100 == 50 and self.current_agents < self.max_agents:
            # Add new agent
            new_position = torch.randn(2, device=self.device) * 0.5
            new_goal = torch.randn(2, device=self.device) * 0.5
            
            success = await self.add_agent_dynamic(new_position, new_goal)
            if success:
                change_log.append({
                    'step': current_step,
                    'action': 'add_agent',
                    'agent_count': self.current_agents
                })
                
        # Remove agents occasionally to demonstrate dynamic capability
        elif current_step % 150 == 0 and self.current_agents > 2:
            # Remove random active agent
            active_indices = self.active_agent_mask.nonzero(as_tuple=True)[0]
            if len(active_indices) > 2:
                remove_id = active_indices[torch.randint(0, len(active_indices), (1,))].item()
                success = await self.remove_agent_dynamic(remove_id)
                if success:
                    change_log.append({
                        'step': current_step,
                        'action': 'remove_agent',
                        'agent_count': self.current_agents
                    })

# Main execution function
async def run_dynamic_simulation():
    """Main function to run the dynamic multi-agent simulation"""
    
    # Configuration
    config = {
        'scenario': 'navigation_v3_test',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_path': 'boundary_constrained_madp_navigation_v3.pth',
        'diffusion_steps': 500,
        'max_agents': 10,
        'initial_agents': 4,
        'horizon_size': 8,
        'total_duration': 1000
    }
    
    # Initialize dynamic manager
    manager = DynamicVMASManager(
        max_agents=config['max_agents'],
        initial_agents=config['initial_agents'],
        horizon_size=config['horizon_size']
    )
    
    # Initialize simulation
    await manager.initialize_simulation(config)
    
    # Load test data (adapted from your validation code)
    from MADP_train_and_sample_v2_2 import NormalizedTrajectoryDataset, collate_fn
    from torch.utils.data import DataLoader
    
    dataset = NormalizedTrajectoryDataset("navigation_v3_Na_4_T_40_dataset.h5", split='test', horizon=40)
    test_loader = DataLoader(dataset, 1, shuffle=True, collate_fn=collate_fn)
    
    frame, start, goal, na, full_traj = next(iter(test_loader))
    frame = frame.to(config['device'])
    
    # Execute dynamic simulation
    results = await manager.vmas_executor.execute_dynamic_moving_horizon(
        manager.diffusion_model,
        frame,
        start,
        goal,
        total_duration=config['total_duration']
    )
    
    # Analysis and reporting
    print(f"\nDynamic Simulation Results:")
    print(f"Total horizons executed: {results['total_horizons']}")
    print(f"Agent changes: {len(results['agent_changes'])}")
    print(f"Final active agents: {results['final_agents']}")
    
    return results

if __name__ == "__main__":
    # Run the dynamic simulation
    results = asyncio.run(run_dynamic_simulation())
