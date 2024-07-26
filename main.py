import torch
import torchvision


def test_cuda():
    print("Checking CUDA availability...")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")


def test_number_of_gpus():
    print("Checking the number of GPUs...")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")


def test_pytorch_version():
    print("Checking PyTorch version...")
    pytorch_version = torch.__version__
    print(f"PyTorch version: {pytorch_version}")


def test_torchvision_version():
    print("Checking torchvision version...")
    torchvision_version = torchvision.__version__
    print(f"torchvision version: {torchvision_version}")


# reality check that the ssh connection is working
if __name__ == "__main__":
    print("SSH connection is working.")
    test_cuda()
    test_number_of_gpus()
    test_pytorch_version()
    test_torchvision_version()
