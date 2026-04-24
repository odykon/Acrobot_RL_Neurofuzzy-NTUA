# Reinforcement Learning on the Acrobot, Comparison of different Model-free approaches

This project applies and compares four reinforcement learning algorithms on the [Acrobot-v1](https://gymnasium.farama.org/environments/classic_control/acrobot/) environment from OpenAI Gymnasium as part of a NeuroFuzzy Control course at NTUA.

## Environment

**Acrobot-v1** is a two-link pendulum task where the goal is to swing the free end of the chain above a fixed threshold. The agent receives a reward of -1 per step until success. The state is a 6-dimensional continuous vector (cosines and sines of the two joint angles, plus their angular velocities). There are 3 discrete actions (apply torque left, none, or right).

## Algorithms Implemented

### 1. SARSA (`SARSA_acrobot.ipynb`)
Tabular on-policy TD control. The continuous state space is discretized into a 6-dimensional grid (10 bins per dimension) using `np.digitize`. Action selection uses an ε-greedy policy with linearly decaying ε. The Q-table is updated with the standard SARSA rule:

```
Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]
```

### 2. Monte Carlo (`MonteCarlo_acrobot.ipynb`)
First-visit Monte Carlo control with the same discretized state representation as SARSA. Full episodes are collected, and returns are computed backwards. Q-values are maintained as running sums normalized by visit count (incremental mean update). ε-greedy policy with linear decay.

### 3. Deep Q-Network — DQN (`DQN_acrobot.ipynb`)
A value-based deep RL method using two neural networks (policy net and target net) with the following key components:
- **Network architecture**: 3-layer MLP (6 → 128 → 64 → 3) with ReLU activations
- **Experience replay**: circular buffer of capacity 10 000, mini-batches of 128
- **Soft target updates**: θ' ← τθ + (1−τ)θ' with τ = 0.005
- **Loss**: Huber loss (SmoothL1) between current Q and Bellman target
- **Optimizer**: AdamW (lr = 1e-4)
- **Exploration**: linearly decaying ε from 1 to 0.01 over 500 episodes

### 4. Policy Gradients — REINFORCE (`PolicyGradients_acrobot.ipynb`)
A policy-based (on-policy) method. A stochastic policy network (6 → 32 → 3) outputs action logits; actions are sampled from the resulting Categorical distribution. The loss is the negative log-probability weighted by the discounted episode return (REINFORCE gradient estimator):

```
L = −E[ G_t · log π(a_t | s_t) ]
```

Training is batched every 1 000 environment steps. Optimizer: Adam (lr = 2e-2).

## Results

Full training curves, evaluation scores, and a detailed comparison of the four algorithms are provided in **Report.pdf**.

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8, 229–256.
- Gymnasium documentation: https://gymnasium.farama.org/environments/classic_control/acrobot/
