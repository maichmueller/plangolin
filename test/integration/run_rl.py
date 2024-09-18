import itertools
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def launch_rl(tmp_path: Path, args: List[str], input_dir: Optional[Path] = None):
    count = 0
    out_dir = tmp_path / "out"
    while out_dir.exists():
        out_dir = tmp_path / f"out_{count}"
        count += 1
    out_dir.mkdir()
    project_root = Path(__file__).parent.parent.parent
    run_py_file = project_root / "experiments" / "rl" / "run.py"
    assert run_py_file.exists() and run_py_file.is_file()
    if input_dir is None:
        input_dir = project_root / "data" / "pddl_domains"

    # split arguments that contain spaces
    args: List[str] = list(
        itertools.chain.from_iterable([arg.split(" ") for arg in args])
    )

    result = subprocess.run(
        [
            sys.executable,
            # use the same python interpreter that is running this script
            str(run_py_file.absolute()),
            "--output_dir",
            str(out_dir),
            "--input_dir",
            str(input_dir),
            "--domain_name",
            "blocks",
            *args,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "blocks" in [f.name for f in out_dir.iterdir() if f.is_dir()]
    out = result.stderr  # logging writes to stderr by default
    out = out.replace("UserWarning: Can't initialize NVML", "")
    assert "UserWarning" not in out
    assert "Finished training" in out
    return result


def test_data_dirs(tmp_path: Path):
    input_dir = "/work/rleap1/jakob.krude/projects/remote/rgnet/data/pddl_domains"
    args = [
        "--instances probBLOCKS-4-0.pddl",
        "--algorithm actor_critic",
        "--embedding_type one_hot",
        "--epochs 1",
        "--logger_backend csv",
    ]
    launch_rl(tmp_path, args, input_dir=Path(input_dir))


def test_actor_critic(tmp_path: Path):
    args = [
        "--instances probBLOCKS-4-0.pddl",
        "--epochs 2",
        "--algorithm actor_critic",
        "--gamma 0.9",
        "--value_net linear",
        "--learning_rate 0.02",
        "--logger_backend csv",
        "--embedding_type gnn",
        "--gnn_hidden_size 8",
    ]
    launch_rl(tmp_path, args)


def test_actor_critic_with_epsilon(tmp_path: Path):
    args = [
        "--instances probBLOCKS-4-0.pddl",
        "--epochs 2",
        "--algorithm actor_critic",
        "--gamma 0.9",
        "--value_net linear",
        "--learning_rate 0.02",
        "--use_epsilon_for_actor_critic True",
        "--logger_backend csv",
        "--embedding_type one_hot",
    ]
    launch_rl(tmp_path, args)


def test_actor_critic_with_all_actions(tmp_path: Path):
    args = [
        "--instances probBLOCKS-4-0.pddl",
        "--epochs 2",
        "--algorithm actor_critic",
        "--gamma 0.9",
        "--value_net linear",
        "--learning_rate 0.02",
        "--use_all_actions True",
        "--logger_backend csv",
        "--embedding_type one_hot",
    ]
    launch_rl(tmp_path, args)


def test_gnn_embedding(tmp_path: Path):
    args = [
        "--instances probBLOCKS-4-0.pddl",
        "--epochs 2",
        "--algorithm actor_critic",
        "--offline True",
        "--embedding_type gnn",
        "--gnn_hidden_size 8",
        "--gnn_aggr softmax",
        "--gnn_num_layer 5",
    ]
    launch_rl(tmp_path, args)


def test_filter_multiple_instances(tmp_path: Path):
    args = [
        "--instances probBLOCKS-4-0.pddl probBLOCKS-4-1.pddl",
        "--epochs 2",
        "--algorithm actor_critic",
        "--offline True",
        "--embedding_type gnn",
        "--gnn_hidden_size 8",
        "--gnn_num_layer 5",
    ]
    result = launch_rl(tmp_path, args)
    assert "Starting training with 2 training problems" in result.stderr


def test_no_instance_filter(tmp_path: Path):
    args = [
        # by omitting --instances, all instances in the directory will be used
        "--epochs 2",
        "--batches_per_epoch 10",
        "--algorithm actor_critic",
        "--offline True",
        "--embedding_type gnn",
        "--gnn_hidden_size 8",
        "--gnn_num_layer 5",
    ]
    result = launch_rl(tmp_path, args)
    assert "Starting training with 3 training problems" in result.stderr


def test_validation(tmp_path: Path):
    args = [
        "--instances probBLOCKS-4-0.pddl probBLOCKS-4-1.pddl",
        "--epochs 2",
        "--algorithm actor_critic",
        # "--offline True",
        "--embedding_type gnn",
        "--gnn_hidden_size 8",
        "--gnn_num_layer 5",
        "--validate_after_epoch True",
    ]
    launch_rl(tmp_path, args)
