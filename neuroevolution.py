# neuroevolution.py

import numpy as np
import gymnasium as gym

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight_dim = input_dim * hidden_dim + hidden_dim * output_dim
        self.reset_weights()

    def reset_weights(self):
        self.weights = np.random.randn(self.weight_dim)

    def set_weights(self, weights):
        self.weights = weights

    def forward(self, x):
        # Unroll weights
        W1 = self.weights[:self.input_dim * self.hidden_dim].reshape(self.input_dim, self.hidden_dim)
        W2 = self.weights[self.input_dim * self.hidden_dim:].reshape(self.hidden_dim, self.output_dim)
        h = np.tanh(x @ W1)
        out = h @ W2
        return out

def evaluate(env_name, weights, input_dim, hidden_dim, output_dim, episodes=3, render=False):
    nn = NeuralNetwork(input_dim, hidden_dim, output_dim)
    nn.set_weights(weights)
    env = gym.make(env_name)
    total_reward = 0.0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = nn.forward(obs)
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = np.argmax(action)
            else:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

    env.close()
    return total_reward / episodes

def run_evolution(env_name, input_dim, hidden_dim, output_dim, pop_size=50, generations=50, elite_frac=0.2, mutation_std=0.1, seed=0):
    np.random.seed(seed)
    elite_size = int(pop_size * elite_frac)
    nn_template = NeuralNetwork(input_dim, hidden_dim, output_dim)
    pop = [np.random.randn(nn_template.weight_dim) for _ in range(pop_size)]
    best_per_gen = []

    for gen in range(generations):
        fitness = [evaluate(env_name, w, input_dim, hidden_dim, output_dim) for w in pop]
        elites = [pop[i] for i in np.argsort(fitness)[-elite_size:]]
        best_per_gen.append(np.max(fitness))
        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            parent = elites[np.random.randint(elite_size)]
            child = parent + mutation_std * np.random.randn(nn_template.weight_dim)
            new_pop.append(child)
        pop = new_pop

    return best_per_gen
