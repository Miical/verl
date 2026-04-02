# Isaac-veRL Disaggregated Training System

## Overview
This project implements a **disaggregated architecture** for training Vision-Language-Action (VLA) models using the veRL framework and Isaac Sim. The primary objective is to support **heterogeneous computing** by splitting heavy rendering tasks from high-throughput training:

* **Training Nodes:** Optimized for high-throughput LLM/VLA training (e.g., NVIDIA H20).
* **Simulation Nodes:** Optimized for rendering and physical simulation in Isaac Sim (e.g., RTX/L-series cards).

---

## Hardware Requirements
To replicate this setup, the following hardware configuration is required:

| Node Type | GPU Configuration | Purpose |
| :--- | :--- | :--- |
| **Training Node** | 8 × NVIDIA H20 | Model training, inference, and rollout management |
| **Simulation Node** | 4 × Rendering-capable GPUs | Isaac Sim environment rendering and physics steps |

---

## Pre-deployment Checklist
Before launching the pipeline, ensure the following modifications and checks are completed:

### 1. Script Configuration
* **Logging & Tracking:** The current version utilizes **Volcengine** (ByteDance) tools for experiment tracking. Update the logging backend if necessary.
* **Storage Paths:** Update paths for high-volume data output, specifically:
    * `replay_pool` save directory.
    * Checkpoint/Model weights output paths.

### 2. Environment & Infrastructure
* **Ray Cluster:** Ensure Ray is correctly installed on all nodes.
* **Consistency:** Verify that both nodes share identical codebases and Python environments (e.g., Docker image or Conda environment).

### 3. Logic Validation
* **Reward Function:** Verify that the reward calculation logic is correctly patched and functioning.
* **Task Mode:** Confirm if the setup is for a **Single Task** (requires specific `task_id`) or the default **Multi-Task** mode.

---

## Deployment Steps

### Step 1: Data Preparation
Generate the training dataset. For single-task training, specify the `task_id`; otherwise, leave it blank for multi-task preparation.

```bash
python prepare_libero_dataset.py --task_id=XX
```

### Step 2: Establish Ray Cluster
Connect the nodes using the Ray framework to enable cross-node resource scheduling. This setup allows the training node to orchestrate the simulation tasks on the simulation node.

**On the Training Node (Head Node):**
Run the following command to initialize the Ray head node with dedicated resources for training, actor, reference, and critic models.

```bash
VERL_LOGGING_LEVEL=INFO python3 -m ray.scripts.scripts start \
    --head --port=6379 \
    --num-gpus=8 --num-cpus=48 \
    --resources='{"train_node": 8, "train_rollout": 8, "train_actor": 8, "train_ref": 8, "train_critic": 8}'
```

**On the Simulation Node (Worker Node):**
Connect the simulation node to the head node. Replace `<HEAD_NODE_IP>` with the actual IP address of your training node.

```bash
python3 -m ray.scripts.scripts start \
    --address='<HEAD_NODE_IP>:6379' \
    --num-cpus=24 \
    --num-gpus=4 \
    --resources='{"sim_node": 4, "sim": 4}'
```

---

### Step 3: Launch Training
Once the Ray cluster is established and both nodes are visible, execute the main training script from the **Training Node**:

```bash
bash verl/verl/experimental/vla/run_pi05_libero_sac_disagg.sh
```

---

### Step 4: Monitoring Output
Rollout visualizations and simulation videos are generated during the training process. Since the simulation tasks are scheduled on the simulation node, the video outputs are stored locally there:

* **Video Output Path:** `/video` (on the Simulation Node)
* **Verification:** Periodically check this directory to ensure the agent's interactions and rendering are functioning as expected.

---

## Troubleshooting
* **Ray Connectivity:** If the nodes fail to connect, ensure that port `6379` and the Ray internal communication ports are open in your network firewall.
* **Resource Mismatch:** Double-check that the `--resources` tags (e.g., `sim_node`) in the `start` command exactly match the resource requirements defined in your Hydra configuration (e.g., `++env.ray_resource_pool_name=sim_node`).
* **Environment Consistency:** Ensure that the Python environment and Isaac Sim version on the simulation node are identical to those on the training node to avoid serialization errors.