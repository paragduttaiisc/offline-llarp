import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from gym.vector.utils import batch_space
from omegaconf import DictConfig
from vc_models.models.vit import model_utils
from transformers import (PhiModel, PhiForCausalLM, PhiConfig)

from llarp.policies.utils import setup_peft_module
from llarp.policies.visual_encoders import VisualEncoderWrapper


class VisualEncoder(VisualEncoderWrapper):
    def __init__(self, use_b16: bool, classifier_feature: str, **kwargs):
        super().__init__()
        (
            self.net,
            self.embd_size,
            self.model_transforms,
            _,
        ) = model_utils.load_model(model_utils.VC1_BASE_NAME)
        self.net.classifier_feature = classifier_feature

        if use_b16:
            self._use_type = torch.bfloat16
        else:
            self._use_type = torch.float32

        self.to(self._use_type)

    def forward(self, img):
        img = img.to(self._use_type)

        img = self.model_transforms(img.permute(0, 3, 1, 2) / 255.0)
        ret = self.net(img)

        if self.net.classifier_feature == "reshape_embedding":
            ret = rearrange(ret, "b d w h -> b (d w h)")
        else:
            ret = rearrange(ret, "b d -> b 1 d")
        assert ret.shape[1:] == self.output_shape

        return ret.to(torch.float32)

    @property
    def output_shape(self):
        if self.net.classifier_feature == "reshape_embedding":
            return np.prod(self.net.patch_embed.grid_size), self.embd_size
        else:
            return 1, self.embd_size



class VisionLanguageBridge(nn.Module):
    def __init__(self, input_size, hidden_size, llm_input_size):
        super().__init__()
        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                input_size,
                hidden_size,
            ),
            nn.ReLU(True),
        )
        self.state_token_proj = nn.Linear(hidden_size, llm_input_size)

    def forward(self, vis_features):
        # There is only 1 visual token. Extract this token and expand.
        if len(vis_features.shape) == 4:
            # Operate on the only visual token.
            assert vis_features.shape[2] == 1

            batch_size = vis_features.shape[0]
            # Flatten and remove #token dim.
            vis_features = rearrange(
                vis_features, "b r 1 d -> (b r) d")
            vis_features = self.visual_fc(vis_features)
            vis_features = rearrange(
                vis_features, "(b r) d -> b r d", b=batch_size)
        else:
            assert vis_features.shape[1] == 1
            vis_features = vis_features[:, 0]

            vis_features = self.visual_fc(vis_features)

        hidden_window = self.state_token_proj(vis_features)
        return hidden_window.unsqueeze(-2)


class EncoderWrapper:
    def __init__(
            self,
            visual_encoder: VisualEncoder,
            llm: PhiForCausalLM,
            lr: float,
            config: DictConfig,
            requires_optimizer: bool = False,
    ) -> None:
        self.visual_encoder = visual_encoder
        self.llm = llm
        self.device = self.llm.device
        visual_encoder_output_shape = visual_encoder.embd_size
        llm_input_size = llm.base_model.model.model.embed_tokens.weight.shape[1]
        self.vl_bridge = VisionLanguageBridge(
            input_size=visual_encoder_output_shape,
            hidden_size=config.hidden_size,
            llm_input_size=llm_input_size
        )
        self.vl_bridge.to(self.device)
        self.optimizer = optim.Adam(self.vl_bridge.parameters(), lr=lr)\
            if requires_optimizer else None

    def llm_embedding_forward(self, prompts):
        return self.llm.model.model.embed_tokens(prompts)

    def vision_encoder_forward(self, obs):
        obs_shape = obs.shape
        obs_embeds = self.visual_encoder(obs.view(-1, *obs_shape[2:]))
        return obs_embeds.view(obs_shape[0], obs_shape[1], -1)

    def vision_language_bridge_forward(self, obs_embeds):
        reshape = False
        batch_size = obs_embeds.shape[0]
        if len(obs_embeds.shape) == 3 and obs_embeds.shape[1] != 1:
            reshape = True
            obs_embeds = rearrange(obs_embeds, "b r d -> (b r) 1 d")
        obs_embeds = self.vl_bridge(obs_embeds)
        if reshape:
            obs_embeds = rearrange(
                obs_embeds, "(b r) 1 d -> b r d", b=batch_size)
        return obs_embeds

    def llm_forward(self, language_embeddings, vision_embeddings):
        hidden = torch.cat([language_embeddings, vision_embeddings], dim=1)
        hidden = hidden.to(torch.bfloat16)  # change to bfloat16
        for layer in self.llm.model.model.layers:
            outputs = layer(hidden)
        outputs = self.llm.model.model.final_layernorm(hidden)
        return outputs.to(torch.float32)

    def forward(self, prompts, observations):
        language_embeddings = self.llm_embedding_forward(prompts)
        if len(observations.shape) != 3:
            vision_embeddings = self.vision_encoder_forward(observations)
        else:
            vision_embeddings = observations.to(torch.float32)
        vl_embeddings = self.vision_language_bridge_forward(vision_embeddings)
        llm_embeddings = self.llm_forward(language_embeddings, vl_embeddings)
        return llm_embeddings[:, 30:, :]

    def save_model(self, path):
        torch.save(self.vl_bridge.state_dict(), path)

    def load_model(self):
        pass


def get_visual_encoder(
        config: DictConfig, device: torch.device = torch.device("cpu")
) -> VisualEncoder:
    visual_encoder = VisualEncoder(**config)
    # freeze the model
    for param in visual_encoder.parameters():
        param.requires_grad = False
    # move to device
    visual_encoder.to(device)

    return visual_encoder


def get_llm(
        llm_id: str,
        device: int = -1,
        use_b16: bool = True,
        load_in_8bit: bool = False,
        use_rope_scaling: bool = False,
        llm_wrapper: DictConfig = None,
        **kwargs,
) -> PhiForCausalLM:
    cmp_llm_id = llm_id.lower()
    kwargs = {}
    if use_b16:
        kwargs = {"torch_dtype": torch.bfloat16}

    if "llama" in cmp_llm_id or "phi" in cmp_llm_id:
        rope_scaling = (
            {"type": "dynamic", "factor": 2} if use_rope_scaling else None
        )
        llm = PhiForCausalLM.from_pretrained(
            llm_id,
            load_in_8bit=load_in_8bit,
            rope_scaling=rope_scaling,
            **kwargs,
        )
        # freeze the model
        for param in llm.parameters():
            param.requires_grad = False
        # PEFT wrapper
        llm = setup_peft_module(
            llm,
            llm_wrapper.peft_full_att_params,
            llm_wrapper.peft_settings
        )
    else:
        raise ValueError(f"Unknown LLM {llm_id}")

    if device >= 0:
        llm.to(device)

    return llm
