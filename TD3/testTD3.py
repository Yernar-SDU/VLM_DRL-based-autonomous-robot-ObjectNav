#!/usr/bin/env python3
import time
import gym
import numpy as np
import torch
from gym import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import the Gazebo environment
from realsense_env import GazeboEnv

############################################
# Environment Wrapper for Stable Baselines3
############################################
class GazeboGymWrapper(gym.Env):
    """
    Wraps the GazeboEnv to conform to Gym's API.
    Converts the tuple (image, scalars) into a dict observation.
    """
    def __init__(self):
        super(GazeboGymWrapper, self).__init__()
        self.env = GazeboEnv()
        # Observation: image is (1, 64, 64), scalars is a 7D vector.
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=1, shape=(1, 64, 64), dtype=np.float32),
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        })
        # Define the action space: e.g. linear [0,1] and angular [-1,1]
        self.action_space = spaces.Box(low=np.array([0, -1.0]),
                                       high=np.array([1.0,  1.0]),
                                       dtype=np.float32)

    def reset(self):
        obs = self.env.reset()  # returns (image, scalars)
        return {"image": obs[0], "scalars": obs[1]}

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        # print('image', next_obs[0])
        return {"image": next_obs[0], "scalars": next_obs[1]}, reward, done, info

############################################
# Custom Feature Extractor for Combined Inputs
############################################
class CombinedExtractor(BaseFeaturesExtractor):
    """
    Fuses CNN features from the image input and MLP features from the scalar input.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super(CombinedExtractor, self).__init__(observation_space, features_dim)
        # CNN for image input
        n_input_channels = observation_space.spaces["image"].shape[0]
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        cnn_output_dim = 64

        # MLP for scalar input (7D)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(observation_space.spaces["scalars"].shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU()
        )
        combined_dim = cnn_output_dim + 64
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(combined_dim, features_dim),
            torch.nn.ReLU()
        )
        self._features_dim = features_dim

    def forward(self, observations):
        img = observations["image"]
        scalars = observations["scalars"]
        cnn_out = self.cnn(img)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        mlp_out = self.mlp(scalars)
        combined = torch.cat([cnn_out, mlp_out], dim=1)
        return self.fc(combined)

def main():
    # Instantiate the environment
    env = GazeboGymWrapper()
    time.sleep(4)  # Allow Gazebo to stabilize

    # Load the trained model (ensure the file is in the same directory or provide the full path)
    model_path = "td3_gazebo_custom_policy.zip"
    model = TD3.load(model_path, env=env)
    print("Loaded trained model from", model_path)

    # Run evaluation episodes with deterministic actions
    num_episodes = 10
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            # Use deterministic policy (no exploration noise)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            time.sleep(0.1)  # adjust sleep time if needed for simulation timing
        print(f"Episode {ep+1} reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
