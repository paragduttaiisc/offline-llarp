import wandb
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from habitat import logger

from llarp.offline.model import EncoderWrapper
from llarp.offline.data_utils import DataBuffer, OfflineDataset


class MomentumContrast:
    def __init__(
            self,
            primary_encoder: EncoderWrapper,
            target_encoder: EncoderWrapper,
            dataset: DataBuffer,
            device: torch.device,
            config: DictConfig
    ) -> None:
        self.primary_encoder = primary_encoder
        self.target_encoder = target_encoder
        self.dataset = dataset
        self.device = device
        self.config = config
        self.can_sample = False
        self.memory_bank = []

    def __append_memory_bank(self, embeddings):
        for embedding in embeddings:
            self.memory_bank.append(embedding)

        if len(self.memory_bank) > self.config.offline.moco.memory_bank_size:
            self.can_sample = True
            self.memory_bank =\
                self.memory_bank[-self.config.offline.moco.memory_bank_size:]

    def __get_negative_samples(self):
        sampled_idxs = torch.randint(
            low=0,
            high=len(self.memory_bank),
            size=(self.config.offline.moco.negative_samples - 1,)
        )
        return torch.stack([self.memory_bank[idx] for idx in sampled_idxs])

    def __train_step(self, i, batch):
        prompts, observations, _, _, _, _, seq_lens = batch

        # Send tensors to device
        prompts = prompts.to(self.device)
        observations = observations.to(self.device)

        embeddings = self.primary_encoder.forward(prompts, observations)
        with torch.no_grad():
            tgt_embeddings = self.target_encoder.forward(prompts, observations)
            tgt_embeddings = tgt_embeddings.detach()

        anchor_embeddings = []
        for embedding, seq_len in zip(embeddings, seq_lens):
            anchor_embeddings.append(embedding[:seq_len])
        anchor_embeddings = torch.cat(anchor_embeddings, dim=0)

        positive_embeddings = []
        for embedding, seq_len in zip(tgt_embeddings, seq_lens):
            positive_embeddings.append(embedding[:seq_len])
        positive_embeddings = torch.cat(positive_embeddings, dim=0)

        self.__append_memory_bank(positive_embeddings)

        if not self.can_sample:
            return

        # positive similarity
        positive_sim = torch.einsum(
            'b d, b d -> b', anchor_embeddings, positive_embeddings
        ).unsqueeze(1)
        # negative similarity
        negative_samples = self.__get_negative_samples()
        negative_sim = torch.einsum(
            'b d, c d -> b c', anchor_embeddings, negative_samples)
        logits = torch.cat([positive_sim, negative_sim], dim=1)

        log_probs = torch.log_softmax(logits, dim=1)

        loss = torch.mean(-log_probs[:, 0])

        self.primary_encoder.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.primary_encoder.vl_bridge.parameters(),
            self.config.offline.max_grad_norm
        )
        self.primary_encoder.optimizer.step()

        # momentum update
        for param, target_param in zip(
            self.primary_encoder.vl_bridge.parameters(),
            self.target_encoder.vl_bridge.parameters()
        ):
            target_param.data.copy_(
                (1 - self.config.offline.moco.momentum) * param.data
                + self.config.offline.moco.momentum * target_param.data)

        if i % 10 == 0:
            logger.info(f"Loss: {loss.item()}")

        if wandb.run is not None:
            wandb.log({"loss": loss.item()})

    def __train_loop(self):
        dataset = OfflineDataset(self.dataset.get_buffer(), self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.offline.batch_size,
            shuffle=True,
            num_workers=4,
        )

        for i, batch in enumerate(dataloader):
            self.__train_step(i, batch)

    def train(self) -> None:
        """
        Train the encoder using momentum contrast for the specified epochs
        """
        num_epochs = self.config.offline.moco.num_epochs
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Starting Epoch {epoch}/{num_epochs}")
            if self.config.offline.dataset.load_vis_embs:
                self.__train_loop()
            else:
                for _ in range(len(self.dataset)):
                    self.__train_loop()
            self.target_encoder.save_model(
                path=self.config.offline.model_save_path+str(self.config.offline.gpu_id)+"/vlb.pt"
            )


class BehavioralCloning:
    def __init__(
            self,
            primary_encoder: EncoderWrapper,
            device: torch.device,
            config: DictConfig
    ) -> None:
        pass

    def __train_step(self):
        pass

    def __train_loop(self):
        pass

    def eval(self):
        pass

    def train(self):
        pass
