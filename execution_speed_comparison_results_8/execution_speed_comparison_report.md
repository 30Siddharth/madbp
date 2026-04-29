# Multi-Agent Trajectory Execution Speed Comparison Report

**Scenario:** navigation_v2
**Number of Agents:** 8
**Test Episodes:** 5
**Max Steps per Episode:** 50

## Executive Summary

This report compares the execution speed and performance of three multi-agent trajectory execution methods:

1. **MAPPO**: Multi-Agent Proximal Policy Optimization with real-time policy inference
2. **MADP**: Multi-Agent Diffusion Policy with moving horizon trajectory generation
3. **CLF-QP**: Control Lyapunov Function with Quadratic Programming (heuristic baseline)

## Performance Summary

| Method   |   Avg Execution Time (s) |   Avg Step Time (ms) |   Steps per Second |   Success Rate (%) |   Avg Reward |   Memory Usage (MB) |   Real-time Factor |   Relative Speed |
|:---------|-------------------------:|---------------------:|-------------------:|-------------------:|-------------:|--------------------:|-------------------:|-----------------:|
| MAPPO    |                 1.36554  |             0.975283 |         1025.34    |                  0 |      1.19365 |             9.2042  |           0.273107 |                1 |
| MADP     |                 0.728449 |            91.0561   |            1.37278 |                  0 |     16.2426  |           160.254   |           0.148663 |                1 |
| CLF-QP   |                 4.52782  |            66.118    |           15.1245  |                  0 |     62.6093  |             8.52256 |           0.905564 |                1 |

## Key Findings

- **Fastest Execution:** MAPPO with 1025.3 steps/second
- **Highest Success Rate:** MAPPO with 0.0%
- **Most Memory Efficient:** CLF-QP with 8.5 MB
- **Real-time Capable Methods:** MAPPO, MADP, CLF-QP

## Methodology

The comparison was conducted by:
1. Generating consistent test scenarios with random start/goal positions
2. Executing each method on identical scenarios
3. Measuring execution time, success rate, and resource usage
4. Analyzing statistical performance across multiple episodes

## Files Generated

- `execution_speed_summary.csv`: Detailed performance metrics
- `execution_speed_comparison.png`: Comprehensive comparison plots
- `timing_analysis_detailed.png`: Detailed timing distribution analysis
