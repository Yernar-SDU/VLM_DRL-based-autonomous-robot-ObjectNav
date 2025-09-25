# td3_lstm/replay_buffer.py

import numpy as np
import torch
import random

class SequenceReplayBuffer:
    def __init__(self, buffer_size, obs_shape, scalar_dim, action_dim, device, seq_len=20):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.full = False
        self.device = device
        self.seq_len = seq_len
        self.max_seq_len = 500  # max episode steps

        self.images = np.zeros((buffer_size, self.max_seq_len, *obs_shape), dtype=np.float32)
        self.scalars = np.zeros((buffer_size, self.max_seq_len, scalar_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, self.max_seq_len, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, self.max_seq_len), dtype=np.float32)
        self.dones = np.zeros((buffer_size, self.max_seq_len), dtype=bool)
        self.lengths = np.zeros((buffer_size,), dtype=int)

    def add_episode(self, image_seq, scalar_seq, action_seq, reward_seq, done_seq):
        idx = self.ptr
        length = len(reward_seq)

        self.images[idx, :length] = image_seq
        self.scalars[idx, :length] = scalar_seq
        self.actions[idx, :length] = action_seq
        self.rewards[idx, :length] = reward_seq
        self.dones[idx, :length] = done_seq
        self.lengths[idx] = length

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.full = self.full or (self.ptr == 0)

    def sample(self, batch_size):
        valid_indices = self.buffer_size if self.full else self.ptr
        batch = []

        for _ in range(batch_size):
            ep_idx = random.randint(0, valid_indices - 1)
            ep_len = self.lengths[ep_idx]

            if ep_len < self.seq_len + 1:
                continue  # skip short episodes

            start = random.randint(0, ep_len - self.seq_len - 1)
            end = start + self.seq_len

            image_seq = self.images[ep_idx, start:end]                      # (T, C, H, W)
            scalar_seq = self.scalars[ep_idx, start:end]                   # (T, D)
            action_seq = self.actions[ep_idx, start:end]                   # (T, A)
            reward_seq = self.rewards[ep_idx, start:end]                   # (T,)
            done_seq = self.dones[ep_idx, start:end]                       # (T,)

            next_image_seq = self.images[ep_idx, start + 1:end + 1]        # (T, C, H, W)
            next_scalar_seq = self.scalars[ep_idx, start + 1:end + 1]     # (T, D)

            batch.append((image_seq, scalar_seq, action_seq, reward_seq, done_seq, next_image_seq, next_scalar_seq))

        # Unpack and convert to tensors
        img, scal, act, rew, don, next_img, next_scal = zip(*batch)

        return (
            torch.tensor(np.array(img), dtype=torch.float32).to(self.device),         # (B, T, C, H, W)
            torch.tensor(np.array(scal), dtype=torch.float32).to(self.device),        # (B, T, D)
            torch.tensor(np.array(act), dtype=torch.float32).to(self.device),         # (B, T, A)
            torch.tensor(np.array(rew), dtype=torch.float32).to(self.device),         # (B, T)
            torch.tensor(np.array(next_img), dtype=torch.float32).to(self.device),    # (B, T, C, H, W)
            torch.tensor(np.array(next_scal), dtype=torch.float32).to(self.device),   # (B, T, D)
            torch.tensor(np.array(don), dtype=torch.float32).to(self.device),         # (B, T)
        )
    def num_sequences(self):
        return self.buffer_size // self.seq_len
    
    def __len__(self):
        return self.buffer_size if self.full else self.ptr
