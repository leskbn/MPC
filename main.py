# %%
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# %%모델 정의
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# %% 데이터 수집
print("[1] 데이터 수집 시작...")
env = gym.make("CartPole-v1")
X, Y = [], []

for ep in range(200):
    state, _ = env.reset()
    done = False
    step = 0
    while not done:
        action = env.action_space.sample()
        next_state, _, done, _, _ = env.step(action)

        X.append(np.append(state, action))
        Y.append([next_state[2]])

        if ep % 50 == 0 and step == 0:
            print(
                f"  예시 샘플: state={state}, action={action}, next_angle={next_state[2]:.4f}"
            )

        state = next_state
        step += 1
print(f"총 수집 샘플 수: {len(X)}")

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

# %%모델 학습
print("[2] 모델 학습 시작...")
model = MLPRegressor(input_dim=5, output_dim=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(1, 101):
    model.train()
    inputs = torch.from_numpy(X)
    targets = torch.from_numpy(Y)

    preds = model(inputs)
    loss = loss_fn(preds, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d} | Loss: {loss.item():.6f}")


# %% MPC 액션 선택 함수
def mpc_select_action(state, model, horizon=3):
    best_action = 0
    best_score = float("-inf")

    print(f"\n[MPC] 현재 상태: angle={state[2]:.4f}")
    for seq in range(2**horizon):
        actions = [(seq >> i) & 1 for i in range(horizon)]
        sim_state = state.copy()
        total_reward = 0

        print(f"  시퀀스 {actions} → 예측 angle:", end=" ")

        for a in actions:
            input_tensor = torch.tensor(
                np.append(sim_state, a), dtype=torch.float32
            ).unsqueeze(0)
            pred_angle = model(input_tensor).item()
            total_reward += -abs(pred_angle)
            sim_state[2] = pred_angle

            print(f"{pred_angle:.4f}", end=" → ")

        print(f"합계 보상: {total_reward:.4f}")

        if total_reward > best_score:
            best_score = total_reward
            best_action = actions[0]

    print(f"👉 선택된 액션: {best_action} (보상: {best_score:.4f})")
    return best_action


# %% 제어 실행
print("\n[3] MPC 제어 시작...")

NUM_EPISODES = 5  # 원하는 에피소드 수
MAX_STEPS = 200  # 한 에피소드에서 최대 스텝 수
env = gym.make("CartPole-v1", render_mode="human")

for ep in range(1, NUM_EPISODES + 1):
    print(f"\n=== Episode {ep} 시작 ===")
    state, _ = env.reset()
    for t in range(1, MAX_STEPS + 1):
        env.render()

        action = mpc_select_action(state, model)
        next_state, _, done, _, _ = env.step(action)

        print(
            f"[Ep {ep:2d} | Step {t:3d}] angle: {state[2]: .4f} → action: {action} → next_angle: {next_state[2]: .4f}"
        )

        if done:
            print(f"  → Episode {ep} 종료: 막대가 {t} 스텝만에 넘어짐!")
            break

        state = next_state
    else:
        print(f"  → Episode {ep} 종료: 최대 스텝 {MAX_STEPS} 도달")

env.close()
