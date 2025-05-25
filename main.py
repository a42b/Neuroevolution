# main.py

from neuroevolution import run_evolution
import matplotlib.pyplot as plt
import gymnasium as gym

def get_env_dims(env_name):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        output_dim = env.action_space.n
    else:
        output_dim = env.action_space.shape[0]
    env.close()
    return input_dim, output_dim

def main(env_name):
    input_dim, output_dim = get_env_dims(env_name)
    hidden_dim = 16
    all_runs = []
    for seed in [0, 1, 2]:
        scores = run_evolution(env_name, input_dim, hidden_dim, output_dim, seed=seed)
        all_runs.append(scores)

    avg_performance = [sum(g)/len(g) for g in zip(*all_runs)]

    plt.plot(avg_performance)
    plt.title(f"Performance on {env_name}")
    plt.xlabel("Generation")
    plt.ylabel("Average Reward")
    plt.grid()
    plt.savefig(f"plots/{env_name}.png")
    plt.close()

if __name__ == "__main__":
    main("CartPole-v1")
    main("MountainCarContinuous-v0")
