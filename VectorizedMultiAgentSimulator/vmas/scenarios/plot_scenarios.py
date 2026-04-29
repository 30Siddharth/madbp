import os
import random
import numpy as np
import matplotlib.pyplot as plt
from vmas import make_env

from vmas import render_interactively




def load_and_capture_scenarios():
    """
    Load three VMAS scenarios, capture one frame from each, and create a visualization plot.
    """
    # Define the three scenarios to load
    scenarios = [
        "navigation_v3",
        "navigation_corridor", 
        "navigation_barrier_v3"
    ]
    # Test each scenario individually
    # render_interactively("navigation_v3", control_two_agents=True)
    # render_interactively("navigation_corridor", control_two_agents=True) 
    # render_interactively("navigation_barrier_v3", control_two_agents=True)
    
    # Dictionary to store captured frames
    captured_frames = {}
    
    # Configuration parameters for consistent visualization
    config = {
        "num_envs": 1,
        "num_agents": 4,
        "device": "cuda",
        "continuous_actions": True,
        "seed": random.randint(0, 1000)  # Random seed for variety
    }
    
    print("Loading and capturing frames from VMAS scenarios...")
    
    for scenario_name in scenarios:
        try:
            print(f"Processing {scenario_name}...")
            
            # Create environment with scenario-specific parameters
            if scenario_name == "navigation_corridor":
                env = make_env(
                    scenario=scenario_name,
                    width=3.0,
                    length=6.0,
                    **config
                )
            elif scenario_name == "navigation_barrier_v3":
                env = make_env(
                    scenario=scenario_name,
                    width=4.0,
                    length=2.0,
                    **config
                )
            else:  # navigation_v3
                env = make_env(
                    scenario=scenario_name,
                    width=3.0,
                    length=1.5,
                    **config
                )
            
            # Reset environment to initialize positions
            obs = env.reset()
            
            # Render one frame as RGB array
            frame = env.render(mode="rgb_array", visualize_when_rgb=True, agent_index_focus=None)
            
            # Store the frame
            captured_frames[scenario_name] = frame
            
            # Clean up environment
            env.close()
            
            print(f"✓ Successfully captured frame from {scenario_name}")
            
        except Exception as e:
            print(f"✗ Error processing {scenario_name}: {str(e)}")
            # Create a placeholder frame if scenario fails to load
            captured_frames[scenario_name] = np.zeros((400, 400, 3), dtype=np.uint8)
    
    return captured_frames

def create_visualization_plot(frames, save_path="vmas_scenarios_comparison.png"):
    """
    Create a side-by-side visualization of the three scenario frames.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    scenario_titles = {
        "navigation_v3": "Navigation Empty\n(Basic Multi-Agent Navigation)",
        "navigation_corridor": "Corridor Navigation\n(Three Vertical Corridors)",
        "navigation_barrier_v3": "Barrier Navigation\n(Central Cross Barrier)"
    }
    
    for idx, (scenario_name, frame) in enumerate(frames.items()):
        axes[idx].imshow(frame)
        axes[idx].set_title(scenario_titles.get(scenario_name, scenario_name), 
                           fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        
        # Add frame border for better visual separation
        for spine in axes[idx].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.suptitle("VMAS Multi-Agent Navigation Scenarios Comparison", 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as: {save_path}")
    
    # Display the plot
    plt.show()
    
    return fig

def save_individual_frames(frames, output_dir="scenario_frames"):
    """
    Save individual frames as separate image files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for scenario_name, frame in frames.items():
        filename = f"{scenario_name}_frame.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save individual frame
        plt.figure(figsize=(8, 6))
        plt.imshow(frame)
        plt.title(f"{scenario_name.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved individual frame: {filepath}")

def main():
    """
    Main execution function to load scenarios and create visualizations.
    """
    print("VMAS Scenario Visualization Tool")
    print("=" * 40)
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    np.random.seed(42)
    
    # Load and capture frames from all scenarios
    frames = load_and_capture_scenarios()
    
    if not frames:
        print("No frames were captured successfully!")
        return
    
    print(f"\nSuccessfully captured {len(frames)} scenario frames")
    
    # Create comparison visualization
    print("\nCreating comparison visualization...")
    fig = create_visualization_plot(frames)
    
    # Save individual frames
    print("\nSaving individual frames...")
    save_individual_frames(frames)
    
    print("\n" + "=" * 40)
    print("Visualization complete!")
    print("Files generated:")
    print("- vmas_scenarios_comparison.png (combined plot)")
    print("- scenario_frames/ (individual frame images)")

if __name__ == "__main__":
    main()
