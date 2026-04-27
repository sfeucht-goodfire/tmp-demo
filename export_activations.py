"""Export neuron activations and down_proj coordinates to JSON for the interactive demo."""

import json
import os

import numpy as np
import torch
from safetensors.torch import load_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTS_DIR = os.path.join(REPO_ROOT, "outputs/neuron_activations/Llama-3.1-8B")
NEURONS_PATH = os.path.join(REPO_ROOT, "notebooks/neurons_per_modulo.json")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

MODEL_SHARD = os.path.expanduser(
    "~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/"
    "snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/"
    "model-00002-of-00004.safetensors"
)
PROBE_DIR = (
    "/mnt/polished-lake/artifacts/fellows-shared/circleDAS/"
    "fourier_probes/summation/resid/sum_ab/layer_18/pos_last"
)

TASKS = ["addition", "months", "weekdays", "hours"]
ACT_TYPES = [("gate", "gate_mlp_acts.pt"), ("up", "up_mlp_acts.pt"), ("both", "both_mlp_acts.pt")]

TEMPLATES = {
    "months": "Q: What month is {number} months after {input}?\nA:",
    "weekdays": "Q: What day is {number} days after {input}?\nA:",
    "hours": (
        "Q: In 24-hour time, it is now {input}:00. "
        "What time will it be in {number} hours?\n"
        "A: In 24-hour time, it will be "
    ),
    "addition": "{input}+{number}=",
}


def export_task_activations(task, all_neurons):
    """Export activation slices for all neurons of interest for one task."""
    task_dir = os.path.join(ACTS_DIR, f"{task}_L18")

    with open(os.path.join(task_dir, "metadata.json")) as f:
        meta = json.load(f)

    result = {"inps": meta["inps"], "nums": meta["nums"], "neurons": {}}

    # Load all three activation tensors
    tensors = {}
    for act_key, act_file in ACT_TYPES:
        t = torch.load(os.path.join(task_dir, act_file), map_location="cpu")
        tensors[act_key] = t  # shape: [num_count, inp_count, 14336]

    for neuron_id in all_neurons:
        neuron_data = {}
        for act_key, _ in ACT_TYPES:
            # Transpose: [num, inp] -> [inp, num] to match heatmap convention (rows=inputs)
            slice_2d = tensors[act_key][:, :, neuron_id].T  # [inp_count, num_count]
            # Round via Python's round() to get clean 2-decimal floats (avoids float32 artifacts)
            neuron_data[act_key] = [
                [round(float(v), 2) for v in row] for row in slice_2d.numpy()
            ]
        result["neurons"][str(neuron_id)] = neuron_data

    out_path = os.path.join(OUTPUT_DIR, f"{task}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  {task}.json: {size_kb:.0f} KB")


def compute_downproj_coordinates(neurons_by_modulo):
    """Compute (cos, sin) Fourier projections for each neuron's down_proj row."""
    print("Loading model down_proj weights...")
    model_tensors = load_file(MODEL_SHARD)
    W_down = model_tensors["model.layers.18.mlp.down_proj.weight"]  # (4096, 14336)

    downproj = {}
    for mod_key, neuron_ids in neurons_by_modulo.items():
        T = int(mod_key.split("_")[1])  # "mod_5" -> 5

        cos_probe_data = torch.load(
            os.path.join(PROBE_DIR, f"probe_mod{T}_cos.pt"), map_location="cpu"
        )
        sin_probe_data = torch.load(
            os.path.join(PROBE_DIR, f"probe_mod{T}_sin.pt"), map_location="cpu"
        )

        d_cos = cos_probe_data["linear.weight"].squeeze()  # (4096,)
        d_sin = sin_probe_data["linear.weight"].squeeze()  # (4096,)
        d_cos = d_cos / d_cos.norm()
        d_sin = d_sin / d_sin.norm()

        downproj[mod_key] = {}
        for nid in neuron_ids:
            w_n = W_down[:, nid].float()
            c = (w_n @ d_cos).item()
            s = (w_n @ d_sin).item()
            downproj[mod_key][str(nid)] = [round(c, 4), round(s, 4)]

        print(f"  {mod_key}: {len(neuron_ids)} neurons projected")

    return downproj


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(NEURONS_PATH) as f:
        neurons_by_modulo = json.load(f)

    all_neurons = sorted(set(n for ns in neurons_by_modulo.values() for n in ns))
    print(f"Exporting {len(all_neurons)} neurons across {len(TASKS)} tasks\n")

    # Export per-task activation JSONs
    print("Exporting activation data:")
    for task in TASKS:
        export_task_activations(task, all_neurons)

    # Compute down_proj coordinates
    print("\nComputing down_proj Fourier projections:")
    downproj = compute_downproj_coordinates(neurons_by_modulo)

    # Write metadata JSON
    neurons_meta = {
        "neurons_by_modulo": neurons_by_modulo,
        "all_neurons": all_neurons,
        "tasks": TASKS,
        "templates": TEMPLATES,
        "downproj": downproj,
    }
    meta_path = os.path.join(OUTPUT_DIR, "neurons.json")
    with open(meta_path, "w") as f:
        json.dump(neurons_meta, f, indent=2)
    print(f"\n  neurons.json: {os.path.getsize(meta_path) / 1024:.0f} KB")

    print("\nDone! Serve with: cd nma/demo && python -m http.server 8080")


if __name__ == "__main__":
    main()
