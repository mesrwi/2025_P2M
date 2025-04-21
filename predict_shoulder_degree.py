import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def signed_angle_from_b_to_a(a, b):
    # 정규화
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    
    # dot과 cross
    dot = np.dot(b_norm, a_norm)
    cross = b_norm[0]*a_norm[1] - b_norm[1]*a_norm[0]  # b → a 방향

    # 부호 각도 계산
    angle_rad = np.arctan2(cross, dot)
    angle_deg = np.degrees(angle_rad)
    
    # 시계 방향이면 +, 반시계 방향이면 -로 뒤집기
    return -angle_deg

def main(args):
    with open(args.json_path, 'r') as f:
        kpt_seq = json.load(f)
    
    theta_list = []
    for kpts in kpt_seq:
        right_shoulder = kpts['keypoints'][6]
        right_elbow = kpts['keypoints'][8]
        right_hip = kpts['keypoints'][12]
        right_ear = kpts['keypoints'][4]
        
        v_shoulder = np.array(right_elbow[:2]) - np.array(right_shoulder[:2])
        v_body = np.array(right_hip[:2]) - np.array(right_ear[:2])
        
        theta_list.append(signed_angle_from_b_to_a(v_shoulder, v_body))
        
    avg_degree = sum(theta_list) / len(theta_list)
    
    print("평균 어깨 각도:", round(avg_degree, 2))
    
    out_dir = os.path.dirname(args.json_path)
    # 시각화 (예: 첫 번째 근육)
    plt.plot(theta_list, label="Calculated shoulder degrees")
    plt.xlabel("Time Step")
    plt.ylabel("Degree")
    plt.title(f"Shoulder Degrees (Average: {round(avg_degree, 2)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/degrees_plot.png")
    print(">> Saved plot to degrees_plot.png")
    
    # 🔽 CSV로 저장
    np.savetxt(f"{out_dir}/shoulder_degrees.csv", np.array(theta_list), delimiter=",")
    print(">> Saved prediction to shoulder_degrees.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    args = parser.parse_args()
    main(args)