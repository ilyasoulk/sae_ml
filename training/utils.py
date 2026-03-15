import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SAEDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if "text" in item:
            return item["text"]

        # hardcoded Aya Dataset Logic
        if "inputs" in item and "targets" in item:
            return item["inputs"] + "\n" + item["targets"]

        raise ValueError(
            f"Dataset item at index {idx} does not contain known text columns. "
            f"Available keys found: {list(item.keys())}"
        )


def get_collate_fn(tokenizer, max_length=128):
    def collate_fn(batch):
        encodings = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

    return collate_fn


class ActivationBuffer:
    def __init__(self, d_model: int, max_size: int = 500_000, device: str = "cuda"):
        self.max_size = max_size
        self.device = device
        self.d_model = d_model

        # Pre-allocate the memory on the GPU once to avoid constant reallocation
        self.buffer = torch.zeros(
            (max_size, d_model), dtype=torch.bfloat16, device=device
        )
        self.current_size = 0

    def add(self, activations: torch.Tensor):
        """Pushes flattened activations into the buffer."""
        num_acts = activations.shape[0]
        if self.current_size + num_acts > self.max_size:
            num_acts = self.max_size - self.current_size
            activations = activations[:num_acts]

        self.buffer[self.current_size : self.current_size + num_acts] = activations
        self.current_size += num_acts

    @property
    def is_full(self) -> bool:
        return self.current_size >= self.max_size

    def drain(self, batch_size: int = 4096):
        """Yields shuffled mini-batches and empties the buffer."""
        indices = torch.randperm(self.current_size, device=self.device)
        for i in range(0, self.current_size, batch_size):
            batch_indices = indices[i : i + batch_size]
            yield self.buffer[batch_indices]
        self.current_size = 0


class HookedActivations:
    """A simple context manager to catch activations from a specific layer."""

    def __init__(self, layer: nn.Module):
        self.activation = None
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        self.activation = hidden_states.detach()

    def remove(self):
        self.hook.remove()


class MultiHookedActivations:
    def __init__(self, target_modules: dict[str, torch.nn.Module]):
        self.activations = {}
        self.handles = []

        for name, module in target_modules.items():

            def make_hook(layer_name):
                def hook_fn(mod, inputs, outputs):
                    self.activations[layer_name] = (
                        outputs[0] if isinstance(outputs, tuple) else outputs
                    ).detach()

                return hook_fn

            self.handles.append(module.register_forward_hook(make_hook(name)))

    def clear(self):
        self.activations = {}

    def remove(self):
        for handle in self.handles:
            handle.remove()


class MultiActivationBuffer:
    def __init__(
        self, layer_names: list[str], d_model: int, max_size: int, device: torch.device
    ):
        self.max_size = max_size
        self.layer_names = layer_names
        self.current_size = 0
        # Initialize parallel buffers
        self.buffers = {
            name: torch.empty((max_size, d_model), dtype=torch.bfloat16, device=device)
            for name in layer_names
        }

    @property
    def is_full(self):
        return self.current_size >= self.max_size

    def add(self, acts_dict: dict[str, torch.Tensor]):
        """acts_dict contains tensors of shape [batch_tokens, d_model]"""
        num_new_tokens = list(acts_dict.values())[0].shape[0]
        add_size = min(num_new_tokens, self.max_size - self.current_size)

        if add_size > 0:
            for name in self.layer_names:
                self.buffers[name][self.current_size : self.current_size + add_size] = (
                    acts_dict[name][:add_size]
                )
            self.current_size += add_size

    def drain(self, batch_size: int):
        indices = torch.randperm(
            self.current_size, device=list(self.buffers.values())[0].device
        )

        for i in range(0, self.current_size, batch_size):
            batch_indices = indices[i : i + batch_size]
            yield {name: self.buffers[name][batch_indices] for name in self.layer_names}

        self.current_size = 0
