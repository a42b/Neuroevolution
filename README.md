# Neuroevolution for Reinforcement Learning

We used a basic neuroevolution algorithm to solve two Gymnasium environments:
- CartPole-v1
- MountainCarContinuous-v0

## Neural Network
- Feedforward with 1 hidden layer (16 units, tanh activation)
- Outputs: 
  - Discrete actions (CartPole): use argmax
  - Continuous actions (MountainCar): clip raw output to env limits

## Evolution Strategy
- Population: 50 individuals
- Generations: 50
- Evaluation: Each individual tested for average reward across 3 episodes
- Selection: Top 20% (elite)
- Mutation: Gaussian noise (std=0.1) applied to elite weights
- Crossover: Not used
- Seeds: [0, 1, 2] (results averaged)

To ensure a robust evaluation, the algorithm was run multiple times with different random seeds:

- Seeds used: 0, 1, and 2
- For each seed:
  - The population evolved for 50 generations
  - The fitness of the best individual was recorded each generation
- Final performance:
  - Averaged the best individual’s score per generation across all 3 seeds

## Result Summary
- **CartPole-v1**: Achieved consistent 500 reward in early generations
- **MountainCarContinuous-v0**: Reached near-optimal reward ~99 within 10 generations

Graphs of average best reward per generation are included in the `plots/` folder.

## How to Run

1. Install dependencies:

```bash
pip install gymnasium numpy matplotlib
```

2. Run experiments:
```bash
python run_experiments.py
```