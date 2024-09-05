import wandb

from llarp.offline.model import (
    EncoderWrapper, PolicyWrapper,get_llm, get_visual_encoder)
from llarp.offline.train_utils import (
    BehavioralCloning, MomentumContrast, BehaviorValueEstimation)
from llarp.offline.data_utils import DataBuffer, DataBufferAlt


def main(cfg):
    device = cfg.habitat_baselines.torch_gpu_id
    config = cfg.habitat_baselines.rl.policy.main_agent.hierarchical_policy\
                .high_level_policy

    # setup wandb
    if cfg.offline.log_to_wandb:
        wandb.init(
            project="cuva",
            entity="statsml-csa-iisc",
            name=f"bve_{cfg.offline.dataset.version}_{cfg.offline.lr}",
            config=cfg,
        )

    # get the pre-trained models
    vis_encoder = get_visual_encoder(config.vis_encoder, device=device)
    llm = get_llm(config.llm_wrapper.llm_id, device=device, **config)

    # build our custom encoder using the pre-trained models
    encoder = EncoderWrapper(
        visual_encoder=vis_encoder,
        llm=llm,
        lr=cfg.offline.lr,
        config=config.visual_bridge,
        requires_optimizer=True
    )
    policy = PolicyWrapper(
        encoder=encoder,
        device=device,
        config=config,
        out_dim=cfg.offline.dataset.num_actions,
        lr=cfg.offline.lr
    )

    # get dataset
    if cfg.offline.dataset.load_vis_embs:
        dataset = DataBufferAlt(cfg.offline)
    else:
        dataset = DataBuffer(cfg.offline)

    if cfg.offline.train_mode == "moco":
        target_encoder = EncoderWrapper(
            visual_encoder=vis_encoder,
            llm=llm,
            lr=cfg.offline.lr,
            config=config.visual_bridge
        )
        # set the target encoder to be the same as the encoder
        target_encoder.vl_bridge.load_state_dict(encoder.vl_bridge.state_dict())

        # pre-train the encoder
        moco_trainer = MomentumContrast(
            primary_encoder=encoder,
            target_encoder=target_encoder,
            dataset=dataset,
            device=device,
            config=cfg
        )
        moco_trainer.train()

    elif cfg.offline.train_mode == "bc":
        bc_trainer = BehavioralCloning(
            policy=policy,
            dataset=dataset,
            device=device,
            config=cfg
        )
        bc_trainer.train()

    elif cfg.offline.train_mode == "bve":
        target_encoder = EncoderWrapper(
            visual_encoder=vis_encoder,
            llm=llm,
            lr=cfg.offline.lr,
            config=config.visual_bridge
        )
        target_policy = PolicyWrapper(
            encoder=target_encoder,
            device=device,
            config=config,
            out_dim=cfg.offline.dataset.num_actions,
            lr=cfg.offline.lr
        )

        bve_trainer = BehaviorValueEstimation(
            policy=policy,
            target_policy=target_policy,
            dataset=dataset,
            device=device,
            config=cfg
        )
        bve_trainer.train()

    else:
        raise NotImplementedError

    wandb.finish()