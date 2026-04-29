# Multi-Agent Trajectory Execution Speed Comparison Report

**Scenario:** navigation_v2
**Number of Agents:** 6
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
| MAPPO    |                 1.01802  |             0.976143 |         1024.44    |                  0 |      4.65983 |             9.04854 |           0.203603 |                1 |
| MADP     |                 0.603848 |            75.481    |            1.65605 |                  0 |      9.28818 |           160.131   |           0.123234 |                1 |
| CLF-QP   |                 3.41559  |            50.5653   |           19.7764  |                  0 |     26.8445  |             8.35391 |           0.683118 |                1 |

## Key Findings

- **Fastest Execution:** MAPPO with 1024.4 steps/second
- **Highest Success Rate:** MAPPO with 0.0%
- **Most Memory Efficient:** CLF-QP with 8.4 MB
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
