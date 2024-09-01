import torch
import pickle
import numpy as np
from habitat import logger
from typing import Tuple, List
from omegaconf import DictConfig
from torch.ao.quantization.backend_config.onednn import observation_type
from torch.utils.data import Dataset
from transformers import CodeGenTokenizer


class DataBuffer:
    def __init__(self, config: DictConfig) -> None:
        self.dataset_path = config.dataset.path + str(config.dataset.version)
        self.gamma = config.gamma
        self.num_splits = 50 if config.dataset.version == 3 else 30
        self._reset_sequence()

    def __len__(self) -> int:
        return self.num_splits

    def _reset_sequence(self) -> None:
        self.permutation = None
        self.permutation_idx = None

    def _get_reward_to_go(
            self,
            rewards: np.ndarray,
            dones: np.ndarray,
    ) -> np.ndarray:
        rtgs = np.zeros_like(rewards)
        rtgs[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            rtgs[t] = rewards[t] + self.gamma * rtgs[t + 1] * (1 - dones[t])
        return rtgs

    def _get_data_buffer(
            self,
            split_idx: int
    ) -> Tuple[
        List[str], np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, List[int]
    ]:
        with open(f'{self.dataset_path}/prompts_{split_idx}.pkl', 'rb') as f:
            prompts = pickle.load(f)
        data = np.load(f'{self.dataset_path}/buffer_{split_idx}.npz')

        images = data['images']
        actions = data['actions']
        rewards = data['rewards']
        dones = data['dones']

        rtgs = self._get_reward_to_go(rewards, dones)
        milestones = [0] + (np.where(dones)[0] + 1).tolist()

        return prompts, images, actions, rewards, dones, rtgs, milestones

    def _set_sequence(self) -> None:
        self.permutation_idx = 0
        self.permutation = np.random.permutation(self.num_splits)

    def get_buffer(self) -> Tuple[
        List[str], np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, List[int]
    ]:
        if self.permutation is None:
            self._set_sequence()

        split_idx = self.permutation[self.permutation_idx] + 1  # 1-indexed

        self.permutation_idx += 1

        if self.permutation_idx >= self.num_splits:
            self._reset_sequence()

        logger.info(
            f"Loading Data Buffer {self.permutation_idx}/{self.num_splits}")
        return self._get_data_buffer(split_idx)

    def __repr__(self) -> str:
        return f"DataBuffer(num_splits={self.num_splits})"


class DataBufferAlt(DataBuffer):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.dataset_path = config.dataset.path + str(config.dataset.version)
        self.gamma = config.gamma
        self.num_splits = 50 if config.dataset.version == 3 else 30
        self.buffer = None
        self.__load_dataset()

    def __load_dataset(self) -> None:
        all_prompts, all_obs, all_acs, all_rews, all_dones, all_rtgs =\
            [], [], [], [], [], []
        for i in range(1, self.num_splits + 1):
            logger.info(f"Loading Data Split {i}/{self.num_splits}")
            with open(f'{self.dataset_path}/prompts_{i}.pkl', 'rb') as f:
                prompts = pickle.load(f)

            data = np.load(f'{self.dataset_path}/buffer_{i}.npz')
            actions = data['actions']
            rewards = data['rewards']
            dones = data['dones']
            rtgs = self._get_reward_to_go(rewards, dones)

            data = np.load(f'{self.dataset_path}/vis_emb_{i}.npz')
            vis_embs = data['vis_embs']

            all_prompts.extend(prompts)
            all_obs.append(vis_embs)
            all_acs.append(actions)
            all_rews.append(rewards)
            all_dones.append(dones)
            all_rtgs.append(rtgs)

        obs = np.concatenate(all_obs, axis=0)
        acs = np.concatenate(all_acs, axis=0)
        rews = np.concatenate(all_rews, axis=0)
        dones = np.concatenate(all_dones, axis=0)
        rtgs = np.concatenate(all_rtgs, axis=0)
        milestones = [0] + (np.where(dones)[0] + 1).tolist()

        self.buffer = (all_prompts, obs, acs, rews, dones, rtgs, milestones)

    def get_buffer(self) -> Tuple[
        List[str], np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, List[int]
    ]:
        if self.buffer is None:
            self.__load_dataset()
        return self.buffer


class OfflineDataset(Dataset):
    def __init__(
            self,
            buffer: Tuple[
                List[str], np.ndarray, np.ndarray, np.ndarray,
                np.ndarray, np.ndarray, List[int]
            ],
            config: DictConfig
    ) -> None:
        self.prompts = buffer[0]
        self.observations = buffer[1]
        self.actions = buffer[2]
        self.rewards = buffer[3]
        self.dones = buffer[4]
        self.rtgs = buffer[5]
        self.milestones = buffer[6]

        self.use_vis_embs = (len(self.observations.shape) == 2)

        self.image_size = config.offline.observation_size
        self.prompt_max_length = config.offline.prompt_max_length
        self.sequence_length = config.offline.sequence_length
        tokenizer_id = config.habitat_baselines.rl.policy.main_agent.\
            hierarchical_policy.high_level_policy.tokenizer_id
        self.tokenizer = CodeGenTokenizer.from_pretrained(tokenizer_id)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __convert_numpy_dtype_to_torch_dtype(
            self,
            dtype: np.dtype
    ) -> torch.dtype:
        if dtype == bool:
            return torch.bool
        elif dtype == np.int8:
            return torch.int8
        elif dtype == np.uint8:
            return torch.uint8
        elif dtype == np.int16:
            return torch.int16
        elif dtype == np.int32:
            return torch.int32
        elif dtype == np.int64:
            return torch.int64
        elif dtype == np.float16:
            return torch.bfloat16
        elif dtype == np.float32:
            return torch.float32
        elif dtype == np.float64:
            return torch.float64
        else:
            raise ValueError(f"Invalid dtype: {dtype}")


    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        start_idx = self.milestones[idx]
        end_idx = self.milestones[idx + 1]

        prompt = self.tokenizer.encode(
            self.prompts[idx], return_tensors="pt", padding="max_length",
            max_length=self.prompt_max_length, return_attention_mask=False
        )[0]

        # initialize tensors
        if self.use_vis_embs:
            observations = torch.zeros(
                (self.sequence_length, self.observations.shape[1]),
                dtype=self.__convert_numpy_dtype_to_torch_dtype(
                    self.observations.dtype))
        else:
            observations = torch.zeros(
                (self.sequence_length, self.image_size, self.image_size, 3),
                dtype=self.__convert_numpy_dtype_to_torch_dtype(
                    self.observations.dtype))
        actions = torch.zeros(
            (self.sequence_length,),
            dtype=self.__convert_numpy_dtype_to_torch_dtype(self.actions.dtype)
        )
        rewards = torch.zeros_like(actions)\
            .to(self.__convert_numpy_dtype_to_torch_dtype(self.rewards.dtype))
        rtgs = torch.zeros_like(rewards)
        dones = torch.zeros_like(rewards)\
            .to(self.__convert_numpy_dtype_to_torch_dtype(self.dones.dtype))

        if self.use_vis_embs:
            observations[:end_idx - start_idx] = torch.from_numpy(
                self.observations[start_idx:end_idx])
        else:
            observations[:end_idx - start_idx] = torch.from_numpy(
                self.observations[start_idx:end_idx].squeeze(1))
        actions[:end_idx - start_idx] =\
            torch.from_numpy(self.actions[start_idx:end_idx])
        rewards[:end_idx - start_idx] =\
            torch.from_numpy(self.rewards[start_idx:end_idx])
        rtgs[:end_idx - start_idx] =\
            torch.from_numpy(self.rtgs[start_idx:end_idx])
        dones[:end_idx - start_idx] =\
            torch.from_numpy(self.dones[start_idx:end_idx])
        seq_len = torch.tensor([end_idx - start_idx], dtype=torch.int32)

        return prompt, observations, actions, rewards, rtgs, dones, seq_len
