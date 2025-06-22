from briscola_env import BriscolaEnv
from stable_baselines3 import PPO

env = BriscolaEnv()

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("briscola_ppo")