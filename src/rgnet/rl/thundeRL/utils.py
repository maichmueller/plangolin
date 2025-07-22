import re
from pathlib import Path

from rgnet.logging_setup import get_logger
from rgnet.rl.data_layout import OutputData


def resolve_checkpoints(
    out_data: OutputData,
) -> tuple[list[Path], Path | None]:
    sorted_checkpoints: list[tuple[int, int, Path]] = []
    last_checkpoint: Path | None = None
    root_dir = out_data.out_dir / "rgnet"
    dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    if len(dirs) != 1:
        get_logger(__name__).warning("Found more than one checkpoint directory.")
    checkpoint_dir = dirs[0] / "checkpoints"
    checkpoint_paths = list(checkpoint_dir.glob("*.ckpt"))
    if len(checkpoint_paths) == 0:
        raise RuntimeError(f"Could not find any checkpoints in {checkpoint_dir}")
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path.stem == "last":
            last_checkpoint = checkpoint_path
        else:
            try:
                match = default_checkpoint_format(checkpoint_path.name)
                epoch, step = match
                sorted_checkpoints.append((epoch, step, checkpoint_path))
            except ValueError:
                get_logger(__name__).warning(
                    f"Skipping checkpoint which was neither called last: {checkpoint_path.name} nor matched default_checkpoint_format"
                )
                continue
    sorted_checkpoints.sort(reverse=True)  # will sort by epoch then by step
    printout = "\n".join(
        f"{epoch = }, {step = }, {path = }"
        for epoch, step, path in map(lambda x: tuple(map(str, x)), sorted_checkpoints)
    )
    get_logger(__name__).info(
        f"Found checkpoints:\n{printout}",
    )
    return [tpl[2] for tpl in sorted_checkpoints], last_checkpoint


def wandb_id_resolver(out_data: OutputData) -> str:
    """
    Try to find the wandb run id from the output directory.
    First look in the wandb directory, where we hope to find the following
    wandb
        run-<time_stamp>-<run_id>
        ...
    Otherwise, we look for the lightning checkpoint directory.
    rgnet
        run_id
            _checkpoints
    """
    wandb_dir = out_data.out_dir / "wandb"
    if wandb_dir.is_dir():
        run_dir = next(wandb_dir.glob("run-*"), None)
        if run_dir is not None:
            run_id = run_dir.name.split("-")[-1]
            if len(run_id) == 8:
                return run_id
    # try the lightning logging directory
    if (lightning_dir := out_data.out_dir / "rgnet").is_dir():
        run_dir = next(lightning_dir.iterdir(), None)
        if run_dir is not None and len(run_dir.name) == 8:
            return run_dir.name
    raise RuntimeError(
        "Could not find a wandb run id in the output directory. "
        "Please ensure that the run was logged with wandb."
    )


def default_checkpoint_format(checkpoint_name: str) -> tuple[int, int]:
    match = re.match(r"epoch=(\d+)-step=(\d+)", checkpoint_name)
    if match is not None:
        epoch, step = map(int, match.groups())
        return epoch, step
    else:
        raise ValueError(
            f"Checkpoint did not follow the pattern 'epoch=<epoch>-step=<step>' got {checkpoint_name}"
        )
