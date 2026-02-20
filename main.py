import numpy as np
import random
import matplotlib.pyplot as plt




class Server:
    def __init__(self, mean_latency, drift):
        self.mean_latency = mean_latency
        self.drift = drift

    def get_latency(self):
        # Drift (non-stationary behavior)
        self.mean_latency += self.drift * (random.random() - 0.5)

        # Gaussian noise
        noise = np.random.normal(0, 1)
        return self.mean_latency + noise



class RoundRobinBalancer:
    def __init__(self, k):
        self.k = k
        self.index = 0

    def select(self):
        choice = self.index
        self.index = (self.index + 1) % self.k
        return choice

    def update(self, i, reward):
        pass


class RandomBalancer:
    def __init__(self, k):
        self.k = k

    def select(self):
        return random.randint(0, self.k - 1)

    def update(self, i, reward):
        pass


class SoftmaxBalancer:
    def __init__(self, k, temperature):
        self.k = k
        self.temperature = temperature
        self.q = np.zeros(k)
        self.counts = np.zeros(k)

    def select(self):
        # Numerical stability: subtract max Q
        max_q = np.max(self.q)
        exp_q = np.exp((self.q - max_q) / self.temperature)
        probs = exp_q / np.sum(exp_q)

        return np.random.choice(self.k, p=probs)

    def update(self, i, reward):
        self.counts[i] += 1
        self.q[i] += (reward - self.q[i]) / self.counts[i]


# ==============================
# SIMULATION
# ==============================

def run_simulation(balancer, servers, steps):
    rewards = []

    for _ in range(steps):
        i = balancer.select()
        latency = servers[i].get_latency()

        reward = -latency  # minimize latency
        balancer.update(i, reward)

        rewards.append(reward)

    return rewards


# ==============================
# MAIN
# ==============================

def main():

    k = 5
    steps = 10000

    # Create non-stationary servers
    servers = [Server(10 + i, drift=0.1) for i in range(k)]

    # Initialize algorithms
    rr = RoundRobinBalancer(k)
    rnd = RandomBalancer(k)
    soft = SoftmaxBalancer(k, temperature=0.5)

    # Run simulations
    rr_rewards = run_simulation(rr, servers, steps)
    rnd_rewards = run_simulation(rnd, servers, steps)
    soft_rewards = run_simulation(soft, servers, steps)

    # Print total reward
    print("Round Robin Total Reward:", sum(rr_rewards))
    print("Random Total Reward:", sum(rnd_rewards))
    print("Softmax Total Reward:", sum(soft_rewards))

    # Plot cumulative reward
    plt.figure()
    plt.plot(np.cumsum(rr_rewards), label="Round Robin")
    plt.plot(np.cumsum(rnd_rewards), label="Random")
    plt.plot(np.cumsum(soft_rewards), label="Softmax")
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Load Balancer Performance Comparison")
    plt.show()


if __name__ == "__main__":

    main()
