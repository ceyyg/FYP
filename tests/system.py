import torch
from src.resnet import ResNet18
from src.data import seed_everything, device
def run_system_tests():
    print("SYSTEM TESTING: ")

    # Reproducibility Test
    seed = 777
    print(f"[Reproducibility] Synchronizing system to seed {seed}...")
    seed_everything(seed)
    m1 = ResNet18().to(device)
    val1 = m1(torch.randn(1, 3, 224, 224).to(device)).detach()

    seed_everything(seed)
    m2 = ResNet18().to(device)
    val2 = m2(torch.randn(1, 3, 224, 224).to(device)).detach()

    match = torch.allclose(val1, val2, atol=1e-5)
    print(f"Stochastic Parity: {match}")
    assert match
    print("System reproducibility confirmed.")