# Установка зависимостей
import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import imageio
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
from tqdm import tqdm

sns.set(style="whitegrid")

print("✅ Библиотеки установлены и импортированы.")

#@title Параметры эксперимента
TOTAL_STEPS = 1_000  #@param {type:"integer"}
NUM_EVAL_EPISODES = 150  #@param {type:"integer"}
FPS = 24  #@param {type:"integer"}
LEARNING_RATE = 1e-4  #@param {type:"number"}
N_STEPS = 512  #@param {type:"integer"}
BATCH_SIZE = 32  #@param {type:"integer"}
N_EPOCHS = 10  #@param {type:"integer"}
GAMMA = 0.98  #@param {type:"number"}
GAE_LAMBDA = 0.94  #@param {type:"number"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

print(f"✅ Параметры заданы. Устройство: {DEVICE}")


def make_env():
    try:
        env = gym.make("Pusher-v5", render_mode="rgb_array")
        return env
    except Exception as e:
        raise RuntimeError(f"Ошибка создания среды: {e}")

try:
    train_env = DummyVecEnv([make_env])
    test_env = make_env()
    print("✅ Среда успешно создана.")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    raise



model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_pusher_tensorboard",
    verbose=1,
    device="auto"
)
print("✅ Модель PPO инициализирована.")


checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

steps_done = 0
chunk_size = 50_000

for chunk in tqdm(range(0, TOTAL_STEPS, chunk_size), desc="Обучение PPO"):
    remaining = min(chunk_size, TOTAL_STEPS - steps_done)
    model.learn(total_timesteps=remaining, reset_num_timesteps=False)
    steps_done += remaining
    
    # Сохранение чекпоинта каждые chunk_size шагов
    checkpoint_path = os.path.join(checkpoint_dir, f"ppo_pusher_{steps_done}.zip")
    model.save(checkpoint_path)
    print(f"💾 Чекпоинт сохранён: {checkpoint_path}")

print("✅ Обучение завершено.")


rewards_per_episode = []
best_reward = -np.inf
best_episode_idx = -1
best_frames = []

for ep in tqdm(range(NUM_EVAL_EPISODES), desc="Оценка агента"):
    obs, _ = test_env.reset()
    frames = []
    total_reward = 0.0
    done = False
    step_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        total_reward += reward
        frames.append(test_env.render())
        step_count += 1

    rewards_per_episode.append(total_reward)

    if total_reward > best_reward:
        best_reward = total_reward
        best_episode_idx = ep
        best_frames = frames

rewards_arr = np.array(rewards_per_episode)
print("-" * 40)
print(f"Награда за лучший эпизод: {best_reward:.2f} (эпизод #{best_episode_idx+1})")
print(f"Средняя награда: {rewards_arr.mean():.2f}")
print(f"Стандартное вознаграждение: {rewards_arr.std():.2f}")
print(f"Минимальная награда: {rewards_arr.min():.2f}, Максимальная награда: {rewards_arr.max():.2f}")


# plt.figure(figsize=(10, 6))
# sns.histplot(rewards_arr, kde=True, color="teal")
# plt.title("Распределение суммарных наград по эпизодам")
# plt.xlabel("Суммарная награда")
# plt.ylabel("Количество эпизодов")
# plt.show()

# # Интерактивный график (Plotly)
# df_rewards = pd.DataFrame({
#     "episode": np.arange(1, len(rewards_per_episode)+1),
#     "reward": rewards_per_episode
# })

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df_rewards["episode"], y=df_rewards["reward"], mode="lines+markers", name="Reward"))
# fig.update_layout(title="Динамика наград по эпизодам", xaxis_title="Эпизод", yaxis_title="Суммарная награда")
# fig.show()


if len(best_frames) > 0:
    video_path = os.path.join(VIDEO_DIR, f"best_episode_{best_episode_idx+1}_reward_{best_reward:.2f}.mp4")
    with imageio.get_writer(video_path, fps=FPS) as writer:
        for frame in best_frames:
            writer.append_data(frame)
    print(f"✅ Видео сохранено: {video_path}")
else:
    print("❌ Нет кадров для сохранения.")
