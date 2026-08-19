from pathlib import Path

from hydra import compose, initialize_config_dir


def test_compose_transformer():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train", overrides=["exp=base", "logger=none", "trainer=1gpu"])
    assert cfg.model._target_.endswith("MotionGenerator")
    assert cfg.denoiser._target_.endswith("MotionTransformerDenoiser")
    assert cfg.model.text_encoder.model_name == f"{cfg.paths.repo_root}/models/t5-base-local"
    assert cfg.model.scheduler.type == "linear_warmup_cosine"
    assert cfg.model.scheduler.warmup_steps == 2000


def test_compose_300m_diffusion_only():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train", overrides=["exp=300m", "logger=none", "trainer=8gpu"])
    assert cfg.denoiser._target_.endswith("MotionTransformerDenoiser")
    assert cfg.loss.simple_root_pos == 0.0
    assert cfg.loss.seam_body_pos == 0.0
    assert cfg.model.use_audio is True
    assert cfg.model.use_human_motion is True
    assert cfg.trainer.devices == 8
    assert cfg.trainer.strategy == "ddp_find_unused_parameters_true"


def test_compose_100m_omnimodal_training():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="train",
            overrides=["exp=100m", "data=omg_data_lerobot_omnimodal", "logger=none", "trainer=4gpu"],
        )
    dataset = cfg.data.dataset_opts.train.omg_lerobot_omnimodal_train
    assert cfg.model.use_audio is True
    assert cfg.model.use_human_motion is True
    assert dataset.use_text is True
    assert dataset.use_audio is True
    assert dataset.use_human_motion is True
    assert dataset.revision == "6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7"


def test_compose_100m_hindi_adapter_core_experiment():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="train",
            overrides=["exp=100m_hindi_adapter", "logger=none", "trainer=1gpu"],
        )
    dataset = cfg.data.dataset_opts.train.bones_seed_hindi_train
    assert cfg.model._target_.endswith("HindiAdapterMotionGenerator")
    assert cfg.model.text_encoder._target_.endswith("MuRILAdapterTextEncoder")
    assert cfg.model.text_encoder.max_length == 50
    assert cfg.model.text_encoder.output_dim == 768
    assert cfg.model.text_encoder.adapter_hidden_dim == 3072
    assert cfg.model.text_encoder.local_files_only is True
    assert cfg.model.text_encoder.model_revision is None
    assert cfg.model.teacher_text_encoder._target_.endswith("FrozenT5TextEncoder")
    assert cfg.model.response_loss_probability == 0.5
    assert cfg.denoiser.hidden_dim == 768
    assert cfg.denoiser.num_layers == 10
    assert cfg.denoiser.num_heads == 12
    assert cfg.denoiser.self_attention_qk_norm is True
    assert cfg.denoiser.cross_attention_qk_norm is True
    assert cfg.model.history_pos_encoding == "sinusoidal"
    assert cfg.model.frame_cond_injection == "per_layer_film"
    assert list(cfg.init_weights_ignore_prefixes) == ["text_encoder."]
    assert cfg.init_weights_legacy_attention_contract == "self-and-cross"
    assert cfg.callbacks.early_stopping.patience == 5
    assert cfg.callbacks.early_stopping.monitor == "val/loss"
    assert dataset._target_.endswith("BonesSeedHindiLeRobotDataset")
    assert dataset.train_window_policy == "random"
