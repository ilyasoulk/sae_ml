import torch
from torch.utils.data import Dataset


class SAEDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["text"]


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
