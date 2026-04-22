import torch
from src.resnet import ResNet18
from src.data import device, data_loaders
from src.optimizer import build_optimizer
from src.checkpoints import save_checkpoint, load_checkpoint
def run_integration_tests_fast():
    print("INTEGRATION TESTING: ")

    # Checkpoint verification
    print("[Checkpoint] Testing weight persistence...")
    m1 = ResNet18().to(device)
    opt1 = build_optimizer("SGD", m1, 0.01, 1e-4)
    save_checkpoint("integ_test", 999, 1, m1, opt1, 0.5, m1.state_dict(), 0)

    m2 = ResNet18().to(device)
    load_checkpoint("integ_test", m2, build_optimizer("SGD", m2, 0.01, 1e-4), 999)
    
    weights_match = all(torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters()))
    print(f"Model Weights Identical after Load: {weights_match}")
    assert weights_match
    print("Checkpoint system verified.\n")

    # Interface
    train_loader, _, _ = data_loaders(42)
    img, lbl, _, _ = next(iter(train_loader))
    print(f"[Data] Batch Shape: {img.shape} | Labels Shape: {lbl.shape} | Device: {device}")
    assert img.shape[1:] == (3, 224, 224)
    print(" Data-to-Model interface verified.\n")