import pathlib

import pytest
import yaml

YAML_DIR = pathlib.Path(__file__).parent / ".." / "functional_tests" / "test_cases"


def get_yaml_files(directory):
    """Retrieve all YAML files from the specified directory."""
    return list([file for file in directory.rglob("model_config.yaml") if file is not None])


def load_yaml(file_path):
    """Load a YAML file and return its content as a Python dictionary."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize(
    "metric",
    ["--log-memory-to-tensorboard", "--log-num-zeros-in-grad", "--log-timers-to-tensorboard"],
)
@pytest.mark.parametrize("yaml_file", get_yaml_files(YAML_DIR))
def test_model_config_tracks_memory(yaml_file, metric):
    """Test if each YAML file contains the required record."""
    print("gpt3-nemo" in str(yaml_file) or "ckpt_converter" in str(yaml_file))
    if any(k in str(yaml_file) for k in ["gpt3-nemo", "ckpt_converter", "gpt-nemo", "inference"]):
        pytest.skip("Skipping `test_model_config_tracks_memory`")

    model_config = load_yaml(yaml_file)

    assert (
        "MODEL_ARGS" in model_config
        and metric in model_config["MODEL_ARGS"]
        and model_config["MODEL_ARGS"][metric] is True
    ), f"Please add argument `{metric}` to `{yaml_file.parent.name}/model_config.yaml` that its metric gets tracked."
