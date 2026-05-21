from wally_ai_search.config.settings import (
    load_dataset_config,
    load_inference_config,
    load_settings,
    load_training_config,
)


def test_load_settings_defaults():
    settings = load_settings()
    assert settings.project_root.is_dir()
    assert settings.dataset_config.name == "dataset.yaml"
    assert settings.training_config.name == "training.yaml"
    assert settings.inference_config.name == "inference.yaml"
    assert settings.log_level == "INFO"


def test_load_yaml_configs():
    dataset = load_dataset_config()
    training = load_training_config()
    inference = load_inference_config()
    assert dataset["nc"] == 1
    assert "model" in training
    assert "weights" in inference
