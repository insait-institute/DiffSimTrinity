# The DiffSim Trinity: Control, Planning, and Search

The DiffSim Trinity contains algorithms based on differentiable simulation for end-to-end control, planning, and search, within the domain of autonomous vehicles. We use [Waymax](https://github.com/waymo-research/waymax) and the Waymo Open Motion Dataset (WOMD). The codebase contains:
- **Analytic World Models (AWM)** for jointly learning a policy and diverse predictive models (odometry, planner, inverse state).
- **Differentiable Simulation Search (DSS)** for gradient-based search at inference-time using learned policies.

This repository is based on the following papers:
- [Autonomous Vehicle Controllers From End-to-End Differentiable Simulation](https://arxiv.org/abs/2409.07965), @IROS 2025
- [Unlocking Efficient Vehicle Dynamics Modeling via Analytic World Models](https://arxiv.org/abs/2502.10012), @AAAI 2026
- [Autonomous Vehicle Path Planning by Searching With Differentiable Simulation](https://arxiv.org/abs/2511.11043), @AAAI 2026

## Repository Layout
- `configs/` – training hyperparameters for each variant plus shared constants.
- `train/` – entrypoints for AWM and DSS training  and the corresponding trainers.
- `eval/` – evaluation entrypoints and metrics loops for AWM and DSS.
- `models/` – feature extractors, state processing utilities, and RNN actor-critic architectures.
- `utils/` – data loading, observations, plotting, and state manipulation.

## Setup
1. Use Python 3.11 with a CUDA-enabled GPU. The packages in `requirements.txt` target CUDA 12.x.
2. (Optional) Create and activate a virtual environment, for example using `micromamba`.
3. Install dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
3. Obtain access to the Waymo Open Motion Dataset and configure Waymax credentials as described in the [Waymax setup guide](https://github.com/waymo-research/waymax?tab=readme-ov-file#configure-access-to-waymo-open-motion-dataset).

### Data paths
Default training/validation paths point to the public GCS URIs:
- Training: `gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/training/training_tfexample.tfrecord@1000`
- Validation: `gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150`

If you mirror the TFRecords locally, update `training_path` / `validation_path` in the relevant config under `configs/`. Control dataset caching via the `dataset_cache` keyword in the config (`True` for in-memory, `False` for none, or a path for on-disk).

## Training
### AWM Agent (policy-driven)
Trains the policy to select actions, while still learning planner and odometry heads.
```bash
python3 train/main_train_awm.py
```
- Uses `configs/conf_train_awm.py`.
- Checkpoints to `logs/train_awm/` by default; change `log_folder` near the top of the script if needed.
- Key toggles: `use_planner_for_train=False`, `num_envs` (batch size).

### AWM Agent (planner-driven)
Uses the planner to select actions during training.
```bash
python3 train/main_train_awm_planner_driven.py
```
- Uses `configs/conf_train_awm_planner_driven.py` (contains the setting `use_planner_for_train=True`).
- Outputs to `logs/train_awm_planner_driven/` by default.

### DSS Agent
At inference time the DSS agent requires two policies, one for the ego-vehicle, which uses route conditioning (last waypoint), and one for the other agents, with no route conditioning.
```bash
# Ego policy (route-conditioned)
python3 train/main_train_search_ego.py

# Other-agents policy (no waypoint conditioning)
python3 train/main_train_search_other.py
```
- Configs live in `configs/conf_train_search_ego.py` and `configs/conf_train_search_other.py`.
- Checkpoints land in `logs/train_search_ego/` and `logs/train_search_other/`.

Adjust batch sizes via `num_envs` if you hit OOM.

## Evaluation
### AWM evaluation
```bash
python3 eval/main_eval_awm.py --expe_id train_awm --epochs 79 \
  --use_planner_for_eval 0 --planning_horizon 10 --num_imagined_rollouts 1
```
- `--expe_id` must match the folder under `logs/`.
- If the model was trained planner-driven, set `--use_planner_for_eval 1`.
- Useful flags: `--use_mpc` to toggle model-predictive control, `--num_imagined_rollouts` (how many rollouts to imagine), `--planning_horizon` (how long to imagine each trajectory), `--top_k` (how many candidate actions to aggregate when using MPC)


### DSS evaluation
Requires both trained policies (ego + others).
```bash
python3 eval/main_eval_search.py --expe_id train_search_other --epochs 39 \
  --ego_policy_weights train_search_ego/params_39.pkl \
  --num_modes 4 --num_actions_to_commit_to 3 \
  --imagination_length 15 --step_size 1000. 0.01
```

### Visualization helpers
`utils/custom_plots.py` contains `plot_animated` and `plot_animated_awm` for visualizing scenarios and imagined trajectories. Call them inside the eval scripts (see inline docstring examples) to dump GIFs of the agent behavior.


## Citations
If you use this work, consider citing:

1. Autonomous Vehicle Controllers From End-to-End Differentiable Simulation
```
@inproceedings{nachkov2024autonomous,
  title={Autonomous Vehicle Controllers From End-to-End Differentiable Simulation},
  author={Nachkov, Asen and Paudel, Danda Pani and Van Gool, Luc},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025},
}
```
2. Unlocking Efficient Vehicle Dynamics Modeling via Analytic World Models
```
@inproceedings{nachkov2025unlocking,
  title={Unlocking Efficient Vehicle Dynamics Modeling via Analytic World Models},
  author={Nachkov, Asen and Paudel, Danda Pani and Zaech, Jan-Nico and Scaramuzza, Davide and Van Gool, Luc},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
3. Autonomous Vehicle Path Planning by Searching With Differentiable Simulation
```
@inproceedings{nachkov2025search,
  title={Autonomous Vehicle Path Planning by Searching With Differentiable Simulation},
  author={Nachkov, Asen and Paudel, Danda Pani and Zaech, Jan-Nico and Scaramuzza, Davide and Van Gool, Luc},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
