## ThundeRL

- Located in `rgnet/rl/thundeRL` and `experiments/rl/thundeRL`
- Optimized RL approach based on two core assumptions:

1. **Small Training Problems**: The problems are small enough to be fully enumerated as
   state space.
2. **One-Step Rollout**: The rollout length is only one step, after which the loss is
   computed.

Under those assumptions we can reduce the problem to a supervised setup utilizing
prefetched data and minimizing host / device memory transfers.

#### Key Differences from General RL Approaches

The core difference is how the training data is gathered and the environment-agent-loss
interaction.

- **Precomputed Rewards and Done Signals**: For each successor state, the reward and
  done signal are precomputed, generating a complete dataset with each state and all its
  successors.
- **Precomputed State Encodings**: The current state and successor states are encoded
  as `HeteroData` graphs, eliminating the need for `pymimir` objects and enabling
  multiprocessing.
- **Reduced Synchronization Overhead**: By sending reward and done signals with each
  state to the device, we avoid the double synchronization
  between `environment -> agent -> environment`.

### General Setup

1. **Dataset Construction**
    - For each training problem, create a dataset (class `FlashDrive`), which can be
      reused across multiple experiments.
    - **State Space Enumeration**: Enumerate the state space and create a data point
      containing

    1. The idx of the state in the state space (saved under `idx_in_space`)
    2. The current stat encoded as graph (`HeteroData`).
    3. The list of successor states encoded as graphs (`HeteroData`).
    4. The reward for each successor state.
    5. The done flag for each successor state.
2. **Training Loop**
    - Iterate over each state of each training problem once per epoch.
    - Use a custom collate function (`thundeRL.collate.collate_fn`) to create batches by
      flattening successor states into a single batch object.
    - Run the agent using `PolicyGradientModule`:
        1. Compute the embeddings of the current states and successor states with
           a `HeteroGNN`
        2. Use the `ActorCritic` agent to compute the transition probabilities and
           action index.
        3. Select the reward and done signal for the sampled action.
        4. Wrap everything in a standard tensordict.
        5. Run the same `ActorCriticLoss` as in the general RL approach to compute the
           loss.
3. **Validation and Testing (Optional)**
    - **Validation Datasets**: Run with custom validation callbacks between training
      runs:
        - `PolicyValidation`: Assess the accuracy of the actor.
        - `CriticValidation`: Evaluate the critic’s difference from optimal values.
        - `ProbsStoreCallback`: Store transition probabilities for each state for later
          analysis.
4. **Testing**: Test the agent on a set of test problems
   using `test_lightning_agent.py`.

#### Additional benefits

- The experiments are fully configurable via yaml files.
    - See [config.yaml](..%2F..%2Ftest%2Fintegration%2Fconfig.yaml) for an example
- Support for all `lightning.Trainer` flags

## Example Usage

```yaml
seed_everything: 42

value_estimator:
  gamma: 0.999

optimizer_setup:
  optimizer:
    class_path: torch.optim.Adam
    init_args:
      lr: 0.0002


data_layout:
  output_data:
    ensure_new_out_dir: true

hetero_gnn:
  hidden_size: 64
  num_layer: 30
  aggr: softmax

trainer:
  accelerator: auto
  devices: 1
  max_time: 00:12:00:00
  val_check_interval: 0.25  # run four times per epoch
  enable_checkpointing: true
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: rgnet
      group: reproduce
  callbacks:
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: 'epoch'

model:
  validation_hooks:
    - class_path: CriticValidation
    - class_path: PolicyValidation
    - class_path: ProbsStoreCallback
      init_args:
        only_run_for_dataloader: [ 0 ]

data:
  batch_size: 128
```

You can separate the data setup in an extra file:

```yaml
data_layout:
  input_data:
    domain_name: blocks
    instances: [ 'probBLOCKS-7-0.pddl', 'probBLOCKS-6-2.pddl', 'probBLOCKS-7-1.pddl', 'probBLOCKS-5-1.pddl', 'probBLOCKS-6-1.pddl', 'probBLOCKS-4-1.pddl', 'probBLOCKS-5-2.pddl', 'probBLOCKS-4-0.pddl', 'probBLOCKS-7-2.pddl', 'probBLOCKS-6-0.pddl' ]
    validation_instances: [ 'probBLOCKS-5-0.pddl', 'probBLOCKS-4-2.pddl' ]

```
