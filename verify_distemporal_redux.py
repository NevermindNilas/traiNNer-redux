import torch
from traiNNer.archs import build_network
import os
import sys

# Mock sys.path to include the traiNNer-redux root
sys.path.append('d:/traiNNer-redux')

def main():
    opt = {
        "type": "distemporal_balanced",
        "num_in_ch": 3,
        "num_out_ch": 3,
        "scale": 4,
        "num_features": 32,
        "num_blocks": 12,
    }

    print("Building network...")
    try:
        net = build_network(opt)
        print("Network built successfully!")

        # Test forward pass with sequence (B, T, C, H, W)
        # PairedVideoDataset returns clips, let's test with T=3
        x = torch.randn(1, 3, 3, 64, 64)
        print(f"Input shape: {x.shape}")

        net.eval()
        with torch.no_grad():
            out = net(x)

        # In eval mode, it returns the whole sequence (B, T, C, H*s, W*s)
        print(f"Output shape (eval): {out.shape}")

        # Test training mode
        net.train()
        out_train = net(x)
        # In training mode, it returns the center frame (B, C, H*s, W*s)
        print(f"Output shape (train): {out_train.shape}")

        print("Verification successful!")
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
