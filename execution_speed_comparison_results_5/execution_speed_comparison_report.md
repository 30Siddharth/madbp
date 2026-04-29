# Multi-Agent Trajectory Execution Speed Comparison Report

**Scenario:** navigation_v2
**Number of Agents:** 5
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
| MAPPO    |                 0.846811 |              1.04155 |          960.103   |                  0 |    -0.964379 |             9.01904 |           0.169362 |                1 |
| MADP     |                 0.542178 |             67.7722  |            1.84441 |                  0 |     7.16056  |           160.048   |           0.110649 |                1 |
| CLF-QP   |                 2.82406  |             42.2011  |           23.6961  |                  0 |    13.8701   |             8.33545 |           0.564812 |                1 |

## Key Findings

- **Fastest Execution:** MAPPO with 960.1 steps/second
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
