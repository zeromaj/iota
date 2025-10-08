import torch
from typing import Optional


def get_available_device() -> str:
    """
    Get the best available device in order of preference:
    CUDA/ROCm > Intel XPU > XLA > DirectML > MPS > Vulkan > PrivateUse1 > CPU

    Note: AMD ROCm GPUs typically use the torch.cuda interface when ROCm is installed.

    Returns:
        str: The device string ('cuda', 'xpu', 'xla', 'dml', 'mps', 'vulkan', 'privateuseone', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"  # This includes both NVIDIA CUDA and AMD ROCm
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    elif torch.is_vulkan_available():
        return "vulkan"
    elif _is_xla_available():
        return "xla"
    elif _is_directml_available():
        return "dml"
    elif hasattr(torch, "privateuseone") and torch.privateuseone.is_available():
        return "privateuseone"
    else:
        return "cpu"


def _is_xla_available() -> bool:
    """Check if XLA backend is available."""
    try:
        import torch_xla  # noqa: F401
        import torch_xla.core.xla_model as xm  # noqa: F401

        return True
    except ImportError:
        return False


def _is_directml_available() -> bool:
    """Check if DirectML backend is available."""
    try:
        import torch_directml  # noqa: F401

        return torch_directml.is_available()
    except ImportError:
        return False


def is_gpu_available() -> bool:
    """Check if any GPU backend is available."""
    return get_available_device() != "cpu"


def get_device() -> torch.device:
    """Get a torch.device object for the best available device."""
    device_str = get_available_device()

    if device_str == "xla":
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401

            return xm.xla_device()
        except ImportError:
            return torch.device("cpu")
    elif device_str == "dml":
        try:
            import torch_directml  # noqa: F401

            return torch_directml.device()
        except ImportError:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def synchronize():
    """Synchronize the current GPU device."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.synchronize()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()
    elif torch.is_vulkan_available():
        # Vulkan synchronization is typically handled automatically
        pass
    elif _is_xla_available():
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401

            xm.wait_device_ops()  # XLA equivalent of synchronize
        except ImportError:
            pass
    elif _is_directml_available():
        try:
            import torch_directml  # noqa: F401

            torch_directml.synchronize()
        except (ImportError, AttributeError):
            pass
    elif hasattr(torch, "privateuseone") and torch.privateuseone.is_available():
        # PrivateUse1 synchronization depends on the backend implementation
        # Some backends may not have synchronize(), so we try-catch
        try:
            torch.privateuseone.synchronize()
        except AttributeError:
            pass
    else:
        pass


def empty_cache():
    """Empty the GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    elif hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()
        return
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
        return
    elif torch.is_vulkan_available():
        # Vulkan memory management is typically automatic
        pass
    elif _is_xla_available():
        # XLA doesn't have explicit cache clearing
        pass
    elif _is_directml_available():
        try:
            import torch_directml  # noqa: F401

            torch_directml.empty_cache()
            return
        except (ImportError, AttributeError):
            pass
    elif hasattr(torch, "privateuseone") and torch.privateuseone.is_available():
        # PrivateUse1 empty_cache depends on the backend implementation
        try:
            torch.privateuseone.empty_cache()
            return
        except AttributeError:
            pass

    # For those that don't have explicit cache we'll do garbage collection instead
    import gc

    gc.collect()


def manual_seed_all(seed):
    """Set the random seed for all available GPU backends."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.manual_seed(seed)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    elif torch.is_vulkan_available():
        # Vulkan uses standard PyTorch seeding
        torch.manual_seed(seed)
    elif _is_xla_available():
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401

            xm.set_rng_state(seed)
        except (ImportError, AttributeError):
            torch.manual_seed(seed)
    elif _is_directml_available():
        try:
            import torch_directml  # noqa: F401

            torch_directml.manual_seed_all(seed)
        except (ImportError, AttributeError):
            torch.manual_seed(seed)
    elif hasattr(torch, "privateuseone") and torch.privateuseone.is_available():
        # PrivateUse1 seeding depends on the backend implementation
        try:
            torch.privateuseone.manual_seed_all(seed)
        except AttributeError:
            try:
                torch.privateuseone.manual_seed(seed)
            except AttributeError:
                torch.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def allocated_memory() -> int:
    """Get the currently allocated GPU memory in bytes."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return torch.mps.current_allocated_memory()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.memory_allocated()
    elif torch.is_vulkan_available():
        # Vulkan memory tracking is not directly available
        return 0
    elif _is_xla_available():
        # XLA memory tracking is not directly available, return 0
        return 0
    elif _is_directml_available():
        try:
            import torch_directml  # noqa: F401

            return torch_directml.memory_allocated()
        except (ImportError, AttributeError):
            return 0
    elif hasattr(torch, "privateuseone") and torch.privateuseone.is_available():
        # PrivateUse1 memory tracking depends on the backend implementation
        try:
            return torch.privateuseone.memory_allocated()
        except AttributeError:
            try:
                return torch.privateuseone.current_allocated_memory()
            except AttributeError:
                return 0
    else:
        return 0  # alt: psutil.virtual_memory().used


def total_memory() -> int:
    """Get the total GPU memory in bytes."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return torch.mps.driver_allocated_memory()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_device_properties(0).total_memory
    elif torch.is_vulkan_available():
        # Vulkan memory info is not directly available
        return int(1e11)  # 100GB fallback
    elif _is_xla_available():
        # XLA memory info is not directly available, return large fallback
        return int(1e11)  # 100GB fallback
    elif _is_directml_available():
        try:
            import torch_directml  # noqa: F401

            return torch_directml.get_device_properties(0).total_memory
        except (ImportError, AttributeError):
            return int(1e11)  # 100GB fallback
    elif hasattr(torch, "privateuseone") and torch.privateuseone.is_available():
        # PrivateUse1 memory info depends on the backend implementation
        try:
            return torch.privateuseone.get_device_properties(0).total_memory
        except AttributeError:
            try:
                return torch.privateuseone.driver_allocated_memory()
            except AttributeError:
                return int(1e11)  # 100GB fallback

    # Fallback to 100GB
    return int(1e11)  # alt: psutil.virtual_memory().total


def available_memory() -> int:
    """Get the available GPU memory in bytes."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return torch.mps.driver_allocated_memory() - torch.mps.current_allocated_memory()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_device_properties(0).total_memory - torch.xpu.memory_allocated()
    elif torch.is_vulkan_available():
        # Vulkan memory calculation is not directly available
        return int(1e11)  # 100GB fallback
    elif _is_xla_available():
        # XLA memory calculation is not directly available
        return int(1e11)  # 100GB fallback
    elif _is_directml_available():
        try:
            import torch_directml  # noqa: F401

            total = torch_directml.get_device_properties(0).total_memory
            allocated = torch_directml.memory_allocated()
            return total - allocated
        except (ImportError, AttributeError):
            return int(1e11)  # 100GB fallback
    elif hasattr(torch, "privateuseone") and torch.privateuseone.is_available():
        # PrivateUse1 memory calculation depends on the backend implementation
        try:
            total = torch.privateuseone.get_device_properties(0).total_memory
            allocated = torch.privateuseone.memory_allocated()
            return total - allocated
        except AttributeError:
            try:
                total = torch.privateuseone.driver_allocated_memory()
                allocated = torch.privateuseone.current_allocated_memory()
                return total - allocated
            except AttributeError:
                return int(1e11)  # 100GB fallback
    else:
        return int(1e11)  # alt: psutil.virtual_memory().total - psutil.virtual_memory().used


def get_device_name(device_index: int = 0) -> Optional[str]:
    """Get the name of the GPU device."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(device_index)
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return "Apple MPS Device"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            return torch.xpu.get_device_name(device_index)
        except AttributeError:
            return "Intel XPU Device"
    elif torch.is_vulkan_available():
        return "Vulkan Device"
    elif _is_xla_available():
        return "XLA Device (TPU/GPU)"
    elif _is_directml_available():
        try:
            import torch_directml  # noqa: F401

            return torch_directml.get_device_name(device_index)
        except (ImportError, AttributeError):
            return "DirectML Device"
    elif hasattr(torch, "privateuseone") and torch.privateuseone.is_available():
        try:
            return torch.privateuseone.get_device_name(device_index)
        except AttributeError:
            return "PrivateUse1 Device"
    else:
        return "CPU"


def get_device_count() -> int:
    """Get the number of available GPU devices."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return 1  # MPS typically has one device
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.device_count()
    elif torch.is_vulkan_available():
        return 1  # Vulkan typically has one device
    elif _is_xla_available():
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401

            return xm.xrt_world_size()
        except (ImportError, AttributeError):
            return 1  # Fallback to 1 device
    elif _is_directml_available():
        try:
            import torch_directml  # noqa: F401

            return torch_directml.device_count()
        except (ImportError, AttributeError):
            return 1  # Fallback to 1 device
    elif hasattr(torch, "privateuseone") and torch.privateuseone.is_available():
        try:
            return torch.privateuseone.device_count()
        except AttributeError:
            return 1  # Fallback to 1 device
    else:
        return 0


def is_amd_gpu() -> bool:
    """Check if the current CUDA device is actually an AMD GPU using ROCm."""
    if not torch.cuda.is_available():
        return False

    try:
        # AMD GPUs using ROCm will have 'AMD' or 'gfx' in the device name
        device_name = torch.cuda.get_device_name(0).lower()
        return "amd" in device_name or "gfx" in device_name or "radeon" in device_name
    except (RuntimeError, AttributeError):
        return False


def is_nvidia_gpu() -> bool:
    """Check if the current CUDA device is an NVIDIA GPU."""
    if not torch.cuda.is_available():
        return False

    try:
        device_name = torch.cuda.get_device_name(0).lower()
        return "nvidia" in device_name or "geforce" in device_name or "tesla" in device_name or "quadro" in device_name
    except (RuntimeError, AttributeError):
        return False


def get_gpu_info() -> dict:
    """Get comprehensive information about the available GPU."""
    info = {
        "available_device": get_available_device(),
        "is_gpu_available": is_gpu_available(),
        "device_count": get_device_count(),
        "device_name": get_device_name(),
        "total_memory_gb": total_memory() / (1024**3),
        "allocated_memory_gb": allocated_memory() / (1024**3),
        "available_memory_gb": available_memory() / (1024**3),
    }

    # Add GPU vendor information if CUDA is available
    if torch.cuda.is_available():
        info["is_nvidia"] = is_nvidia_gpu()
        info["is_amd_rocm"] = is_amd_gpu()
        info["cuda_version"] = torch.version.cuda if torch.version.cuda else "Unknown"

    return info


def print_gpu_info():
    """Print detailed GPU information."""
    info = get_gpu_info()
    print("=== GPU Information ===")
    print(f"Available Device: {info['available_device']}")
    print(f"GPU Available: {info['is_gpu_available']}")
    print(f"Device Count: {info['device_count']}")
    print(f"Device Name: {info['device_name']}")
    print(f"Total Memory: {info['total_memory_gb']:.2f} GB")
    print(f"Allocated Memory: {info['allocated_memory_gb']:.2f} GB")
    print(f"Available Memory: {info['available_memory_gb']:.2f} GB")

    if "is_nvidia" in info:
        print(f"NVIDIA GPU: {info['is_nvidia']}")
        print(f"AMD ROCm GPU: {info['is_amd_rocm']}")
        print(f"CUDA Version: {info['cuda_version']}")

    print("========================")


if __name__ == "__main__":
    # Test the GPU detection and information functions
    print_gpu_info()

    # Test individual functions
    print("\nTesting individual functions:")
    print(f"Available device: {get_available_device()}")
    print(f"GPU available: {is_gpu_available()}")
    print(f"Device count: {get_device_count()}")
    print(f"Device name: {get_device_name()}")

    # Test memory functions (should not crash even without GPU)
    print(f"Total memory: {total_memory() / (1024**3):.2f} GB")
    print(f"Allocated memory: {allocated_memory() / (1024**3):.2f} GB")
    print(f"Available memory: {available_memory() / (1024**3):.2f} GB")

    # Test synchronization and cache clearing (should not crash)
    print("\nTesting GPU operations:")
    synchronize()
    print("Synchronize: OK")
    empty_cache()
    print("Empty cache: OK")
    manual_seed_all(42)
    print("Manual seed: OK")

    print("\nAll tests completed successfully!")
