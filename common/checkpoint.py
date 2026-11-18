from typing import Any
import torch
import os


class CheckpointError(RuntimeError):
    """Base class for explicit checkpoint save/load failures."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised before mutation when a checkpoint contract is incompatible."""

class Checkpointer:
    def __init__(self, ckpt_path = None):
        self.ckpt_path = ckpt_path
        self.models = {}
        self.ckpt = {}

    def save(self, path_override = None, strict=False):
        path = path_override or self.ckpt_path
        try:
            os.makedirs(os.path.dirname(path))
        except FileExistsError:
            pass
        models = {}
        for k, v in self.models.items():
            if strict:
                if hasattr(v, "state_dict"):
                    models[k] = v.state_dict()
                elif hasattr(v, "save"):
                    models[k] = v.save()
                else:
                    raise CheckpointError(
                        f"registered object {k!r} is not serializable in strict mode"
                    )
                continue
            try:
                models[k] = v.state_dict()
            except:
                try:
                    models[k] = v.save()
                except:
                    models[k] = v
        torch.save(models, path)
    
    def register(self, models):
        self.models.update(models)

    def load(
            self,
            path_override=None,
            filter_keys=None,
            strict=False,
            required_keys=None,
            map_location=None,
    ):
        path = path_override or self.ckpt_path
        if not os.path.exists(path):
            print(f"no checkpoint at {path} found")
            return False
        print(f"loading checkpoint from {path}")
        self.ckpt = torch.load(path, map_location=map_location)
        if strict:
            keys = list(filter_keys) if filter_keys is not None else list(self.models)
            required = set(keys if required_keys is None else required_keys)
            missing_checkpoint = required.difference(self.ckpt)
            missing_registered = required.difference(self.models)
            if missing_checkpoint:
                raise CheckpointCompatibilityError(
                    "checkpoint is missing required keys: "
                    + ", ".join(sorted(missing_checkpoint))
                )
            if missing_registered:
                raise CheckpointCompatibilityError(
                    "loader is missing registered keys: "
                    + ", ".join(sorted(missing_registered))
                )

            # Contract-like objects can reject a checkpoint before model,
            # optimizer, step, or RNG state is mutated.
            for key in keys:
                target = self.models[key]
                if hasattr(target, "validate"):
                    target.validate(self.ckpt[key])

            for key in keys:
                target = self.models[key]
                value = self.ckpt[key]
                try:
                    if isinstance(target, torch.nn.Module):
                        incompatible = target.load_state_dict(value, strict=True)
                        if incompatible.missing_keys or incompatible.unexpected_keys:
                            raise CheckpointCompatibilityError(
                                f"state dict mismatch for {key!r}: {incompatible}"
                            )
                    elif hasattr(target, "load_state_dict"):
                        target.load_state_dict(value)
                    elif hasattr(target, "load"):
                        target.load(value)
                    else:
                        raise CheckpointCompatibilityError(
                            f"registered object {key!r} cannot be restored strictly"
                        )
                except CheckpointError:
                    raise
                except Exception as exc:
                    raise CheckpointCompatibilityError(
                        f"failed to restore checkpoint key {key!r}: {exc}"
                    ) from exc
                print(f"successfully loaded {key} (strict)")
            return True

        for k, v in self.ckpt.items():
            if (filter_keys is not None) and (k not in filter_keys):
                continue
            try:
                self.models[k].load_state_dict(v)
                print(f"successfully loaded state dict for {k}!!!")
            except:
                try:
                    self.models[k].load(v)
                    print(f"successfully loaded {k}")
                except:
                    self.models[k] = v
                    print(f"successfully loaded {k}")
        return True
