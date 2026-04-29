# Multi-Agent Trajectory Execution Speed Comparison Report

**Scenario:** navigation_v2
**Number of Agents:** 2
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
| MAPPO    |                 0.401893 |             0.924571 |         1081.58    |                  0 |     0.594356 |             8.88955 |          0.0803786 |                1 |
| MADP     |                 0.38339  |            47.9237   |            2.60831 |                  0 |     0.906812 |           159.975   |          0.0782428 |                1 |
| CLF-QP   |                 1.13423  |            16.7525   |           59.6927  |                  0 |     4.26607  |             8.32246 |          0.226847  |                1 |

## Key Findings

- **Fastest Execution:** MAPPO with 1081.6 steps/second
- **Highest Success Rate:** MAPPO with 0.0%
- **Most Memory Efficient:** CLF-QP with 8.3 MB
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
