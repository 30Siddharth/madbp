import torch
import numpy as np
from MADP_validate import execute_hierarchical_moving_horizon_system

def run_simple_scenario_test():
    """
    Simple script that calls the main execution function from MADP_validate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Running Simple Scenario Test")
    print("="*40)
    
    # Call the main execution function from MADP_validate
    results, metrics = execute_hierarchical_moving_horizon_system(
        scenario_name="navigation_v2_test",
        model_path="boundary_constrained_madp_navigation_v2_na_4.pth",
        device=device,
        goal_tolerance=0.1,
        goal_pursuit_threshold=0.25,
        horizon_size=8
    )
    
    if results is not None:
        print("\n✅ Test completed successfully!")
        print(f"Success Rate: {metrics['success_rate']:.2f}")
        print(f"Real-time Capable: {metrics['real_time_capable']}")
        print(f"Total Steps: {metrics['total_steps']}")
    else:
        print("❌ Test failed!")
    
    return results, metrics

def run_custom_scenario_test():
    """
    Run a test with custom start and goal positions
    """
    from MADP_validate import HierarchicalMovingHorizonController
    from MADP_diffusion_v2_2 import EnhancedMultiAgentDiffusionModel
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define custom positions
    custom_starts = np.array([
        [0.5, 0.5],
        [-0.5, 0.5],
        [0.5, -0.5],
        [-0.5, -0.5]
    ])
    
    custom_goals = np.array([
        [-0.5, -0.5],
        [0.5, -0.5],
        [-0.5, 0.5],
        [0.5, 0.5]
    ])
    
    print("Running Custom Scenario Test")
    print(f"Custom starts: {custom_starts}")
    print(f"Custom goals: {custom_goals}")
    
    # Load model
    model = EnhancedMultiAgentDiffusionModel(
        max_agents=10,
        horizon=8,
        state_dim=2,
        img_ch=3,
        hid=128,
        diffusion_steps=150,
        schedule_type='linear'
    ).to(device)
    
    model.load_state_dict(torch.load("boundary_constrained_madp_navigation_v2_na_4.pth", map_location=device))
    model.eval()
    
    # Initialize controller
    controller = HierarchicalMovingHorizonController(
        scenario_name="navigation_v2_test",
        num_envs=1,
        max_steps=200,
        n_agents=4,
        device=device,
        test_start_positions=custom_starts,
        test_goal_positions=custom_goals,
        horizon_size=8,
        goal_tolerance=0.1,
        goal_pursuit_threshold=0.25
    )
    
    # Create dummy frame and position tensors
    frame = torch.zeros(1, 3, 600, 800, device=device)
    starts = torch.tensor(custom_starts, device=device).unsqueeze(0)
    goals = torch.tensor(custom_goals, device=device).unsqueeze(0)
    n_agents = torch.tensor([4], device=device)
    
    # Execute test
    results = controller.execute_hierarchical_trajectory(
        model=model,
        frames=frame,
        starts=starts,
        goals=goals,
        n_agents=n_agents,
        full_horizon_length=50,
        render=True,
        store_frames=True
    )
    
    # Create outputs
    controller.create_gif_from_frames(results, "custom_scenario_test.gif")
    controller.plot_comprehensive_analysis(results, "custom_scenario_analysis.png")
    
    # Evaluate and print results
    metrics = controller.evaluate_performance(results)
    
    print(f"\nCustom Scenario Results:")
    print(f"Success Rate: {metrics['success_rate']:.2f}")
    print(f"Mean Tracking Error: {metrics['mean_tracking_error']:.4f}")
    print(f"Real-time Capable: {metrics['real_time_capable']}")
    
    return results, metrics

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Simple scenario test (calls execute_hierarchical_moving_horizon_system)")
    print("2. Custom scenario test (uses HierarchicalMovingHorizonController directly)")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        results, metrics = run_simple_scenario_test()
    elif choice == "2":
        results, metrics = run_custom_scenario_test()
    else:
        print("Invalid choice")
