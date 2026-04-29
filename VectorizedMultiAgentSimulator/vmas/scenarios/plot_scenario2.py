import os
import numpy as np
import matplotlib.pyplot as plt
from vmas import make_env
import torch

def capture_working_frame(scenario_name, config):
    """Capture frame with forced rendering initialization"""
    try:
        print(f"Loading {scenario_name}...")
        
        # Create environment
        env = make_env(scenario=scenario_name, **config)
        
        # Critical: Reset and take multiple steps to initialize rendering
        obs = env.reset()
        
        # Take several random steps to ensure environment is fully initialized
        for _ in range(5):
            # random_actions = [env.action_space.sample() for _ in range(env.n_agents)]
            # import pdb
            # pdb.set_trace()
            # random_actions = torch.randn(1, config["n_agents"], 2, device=env.device) * 0.1
            # random_actions = torch.tensor([[[ 0.0807,  0.0665],      
            #                     [ 0.0746,  0.1481],         
            #                     [-0.0837, -0.0172],         
            #                     [ 0.0235,  0.0986]]]) 

            random_actions = [
                        0.1*torch.randn(1, 2),  # Agent 0: shape (num_envs, action_dim)
                        0.1*torch.randn(1, 2),  # Agent 1: shape (num_envs, action_dim)  
                        0.1*torch.randn(1, 2),  # Agent 2: shape (num_envs, action_dim)
                        0.1*torch.randn(1, 2)   # Agent 3: shape (num_envs, action_dim)
                    ]       

            obs, rewards, dones, info = env.step(random_actions)
        
        # Force rendering initialization by calling render multiple times
        for attempt in range(3):
            frame = env.render(mode="rgb_array", visualize_when_rgb=True)
            
            # Check if frame has actual content
            if frame is not None and frame.size > 0 and not np.all(frame == 0):
                print(f"✓ Successfully captured frame for {scenario_name}")
                env.reset()
                return frame
            
            # Take another step and try again
            if attempt < 2:
                random_actions = [env.action_space.sample() for _ in range(env.n_agents)]
                env.step(random_actions)
        
        print(f"⚠️ {scenario_name} produced blank frame, using fallback")
        env.reset()
        return create_debug_frame(scenario_name)
        
    except Exception as e:
        print(f"❌ Error with {scenario_name}: {str(e)}")
        return create_debug_frame(scenario_name)

def create_debug_frame(scenario_name):
    """Create a colored debug frame to identify the scenario"""
    colors = {
        "navigation_v3": [100, 150, 200],
        "navigation_corridor_v2": [150, 100, 200], 
        "navigation_barrier_v3": [200, 150, 100]
    }
    
    frame = np.ones((400, 400, 3), dtype=np.uint8)
    color = colors.get(scenario_name, [128, 128, 128])
    frame[:, :] = color
    
    return frame

def load_scenarios_with_fallback():
    """Load scenarios with comprehensive fallback strategy"""
    
    scenarios_config = {
        "navigation_v3": {
            "num_envs": 1,
            "n_agents": 4,
            "device": "cuda",
            "continuous_actions": True,
            "collisions": True,
            "shared_rew": True,
            "width": 4.0,
            "length": 2.0,
        },
        "navigation_corridor_v2": {
            "num_envs": 1,
            "n_agents": 4,
            "device": "cuda",
            "continuous_actions": True,
            "width": 3.0,
            "length": 3.0,
            "collisions": True
        },
        "navigation_barrier_v3": {
            "num_envs": 1,
            "n_agents": 4,
            "device": "cuda",
            "continuous_actions": True,
            "collisions": True
        }
    }
    
    captured_frames = {}
    
    for scenario_name, config in scenarios_config.items():
        frame = capture_working_frame(scenario_name, config)
        captured_frames[scenario_name] = frame
    
    return captured_frames

def create_enhanced_visualization(frames, save_path="vmas_scenarios_fixed.png"):
    """Create visualization with enhanced debugging information"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # fig.suptitle("Multi-Agent Navigation Scenarios", fontsize=20, fontweight='bold')
    
    scenario_titles = {
        "navigation_v3": "Navigation Empty\n(Basic Multi-Agent)",
        "navigation_corridor_v2": "Corridor Navigation\n(Vertical Corridors with Obstacles)",
        "navigation_barrier_v3": "Barrier Navigation\n(Central Cross Barrier)"
    }
    
    for idx, (scenario_name, frame) in enumerate(frames.items()):
        axes[idx].imshow(frame)
        axes[idx].set_title(scenario_titles.get(scenario_name, scenario_name), 
                           fontsize=18)
        axes[idx].axis('off')
        
        # Add frame info for debugging
        frame_info = f"Shape: {frame.shape}\nNon-zero: {np.count_nonzero(frame)}"
        # axes[idx].text(0.02, 0.98, frame_info, transform=axes[idx].transAxes,
        #               fontsize=8, verticalalignment='top', 
        #               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=96, bbox_inches='tight')
    print(f"Enhanced visualization saved as: {save_path}")
    plt.show()
    
    return fig

def main():
    """Main execution with comprehensive error handling"""
    print("VMAS Scenario Visualization - Enhanced Version")
    print("=" * 60)
    
    # Load scenarios with fallback
    frames = load_scenarios_with_fallback()
    
    # Create enhanced visualization
    create_enhanced_visualization(frames)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")

if __name__ == "__main__":
    main()
