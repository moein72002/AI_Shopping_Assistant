import torch

def check_gpu():
    """Checks if PyTorch can detect and use the GPU."""
    print(f"PyTorch version: {torch.__version__}")
    is_available = torch.cuda.is_available()
    print(f"Is CUDA available? {is_available}")
    if is_available:
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device index: {current_device}")
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Current CUDA device name: {device_name}")
    else:
        print("CUDA is not available. PyTorch is running on CPU.")

if __name__ == '__main__':
    check_gpu()
