# Softmax-Based Client-Side Load Balancer

## 📌 Project Overview

This project implements a client-side load balancing strategy for a distributed system consisting of **K non-stationary servers** with noisy latency.

The objective is to **minimize total latency** (equivalently maximize cumulative reward).

Unlike traditional static approaches such as Round Robin or Random selection, this project implements **Softmax Action Selection**, a probabilistic learning-based method inspired by the Multi-Armed Bandit problem.

---

## 🧠 Problem Definition

Each server:

- Has a time-varying (non-stationary) mean latency
- Contains Gaussian noise
- Simulates real-world distributed system uncertainty

The environment changes over time, meaning static algorithms cannot adapt.

This problem can be modeled as a **Non-Stationary Multi-Armed Bandit** problem.

---

## ⚙️ Implemented Algorithms

### 1️⃣ Round Robin
- Cycles through servers sequentially
- No learning
- No adaptation

### 2️⃣ Random Selection
- Selects servers randomly
- No learning
- No adaptation

### 3️⃣ Softmax Action Selection
- Maintains estimated reward values (Q-values)
- Selects servers probabilistically:

\[
P(i) = \frac{e^{Q_i / T}}{\sum_j e^{Q_j / T}}
\]

Where:
- Q_i = estimated reward of server i
- T = temperature parameter controlling exploration-exploitation tradeoff

---

## 🔥 Why Softmax?

Softmax enables:

- Adaptive learning
- Exploration-exploitation balance
- Probabilistic decision making
- Better performance in dynamic environments

Unlike Round Robin and Random, Softmax uses historical performance data.

---

## 🧮 Numerical Stability

Direct exponential computation can cause overflow:

\[
e^{Q}
\]

To prevent this, the implementation subtracts the maximum Q-value before exponentiation:

\[
e^{(Q - max(Q))}
\]

This technique is known as the **Log-Sum-Exp trick**, ensuring numerical stability.

---

## ⏱ Time Complexity Analysis

For each decision step:

- Max Q computation → O(K)
- Exponentiation → O(K)
- Probability normalization → O(K)

Total per step complexity:

\[
O(K)
\]

Overall simulation complexity:

\[
O(T \times K)
\]

Where:
- T = number of time steps
- K = number of servers

---

## 📊 Results

Simulation results show that:

- Softmax outperforms Round Robin and Random
- It adapts to performance drift
- It achieves higher cumulative reward over time

Graph visualization demonstrates this performance difference.

---

## 🚀 How to Run

Install dependencies:

```bash
pip install numpy matplotlib
