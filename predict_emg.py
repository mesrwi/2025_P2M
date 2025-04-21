import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from net.legacy_model import P2M, P2M_MultiHead
from config import get_config

def main(args):
    cfg = get_config(args.config)
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드 + eval 모드
    model = eval(cfg.model.model).load_from_checkpoint(args.ckpt_path, args=cfg)
    model.eval()
    model.to(device)

    # 데이터 로드
    x = np.load(args.npy_path)  # shape: (T, N, 3)

    # 텐서 변환 및 차원 변경
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, T, N, 3)
    x_tensor = x_tensor.to(device)

    # 조건값 설정
    if condval:
        condval = torch.tensor([args.condval], dtype=torch.float32).to(device)  # (1,)

        # 예측
        with torch.no_grad():
            pred = model(x_tensor, condval)  # (1, T, num_targets)
    else:
        with torch.no_grad():
            pred = model(x_tensor)

    # 출력
    pred_np = pred[0].squeeze(-1).cpu().numpy()  # (T,)
    print("예측 결과 shape:", pred_np.shape)
    
    out_dir = os.path.dirname(args.npy_path)
    
    # 시각화 (예: 첫 번째 근육)
    plt.plot(pred_np, label="Predicted Muscle Activation")
    plt.xlabel("Time Step")
    plt.ylabel("Activation")
    plt.title("Muscle Activation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/activation_plot.png")
    print(">> Saved plot to activation_plot.png")
    
    # 🔽 CSV로 저장
    np.savetxt(f"{out_dir}/predicted_activation.csv", pred_np, delimiter=",")
    print(">> Saved prediction to predicted_activation.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=str, required=True)
    parser.add_argument("--condval", type=float)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()
    main(args)