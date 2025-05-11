import torch
import numpy as np
import librosa
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('../Train')) # 사용한 models.py를 import하기 위해서


# ----------------------------
# 설정
# ----------------------------
SEGMENT_TYPE = "short"
EPOCH = 1000
BATCH = 1000
MODEL_PATH = f"../Train/trained_model/ESC50_CNN_{SEGMENT_TYPE}_{EPOCH}_{BATCH}.pt"

VOTING = "probability" 
#VOTING = "majority"
AUDIO_DIR = "./TestAudio"

SR = 22050
N_MELS = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

# ----------------------------
# category map 불러오기
# ----------------------------
def load_category_map(csv_path="../Train/ESC-50/meta/esc50.csv"):
    meta_df = pd.read_csv(csv_path)
    return meta_df.drop_duplicates('target')[['target', 'category']] \
                  .set_index('target')['category'].to_dict()

category_map = load_category_map()

# ----------------------------
# 모델 로딩
# ----------------------------
segment_frames = 101 if SEGMENT_TYPE == "long" else 41
input_shape = (2, N_MELS, segment_frames)

model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.to(DEVICE)
model.eval()

# ----------------------------
# 오디오 → 멜스펙트로그램 변환
# ----------------------------
def preprocess_audio_to_segments(audio_path, segment_frames=101, overlap=0.9):
    y, _ = librosa.load(audio_path, sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    log_mel = librosa.power_to_db(S=mel)
    delta = librosa.feature.delta(data=log_mel)
    mel_delta = np.stack([log_mel, delta], axis=0)

    step = int(segment_frames * (1 - overlap))
    segments = []
    for i in range(0, mel_delta.shape[2] - segment_frames + 1, step):
        seg = mel_delta[:, :, i:i+segment_frames]
        if seg.shape[2] == segment_frames:
            segments.append(seg)
    return segments

# ----------------------------
# 예측 함수 (확률 포함 반환)
# ----------------------------
def predict(audio_path, gt_target):
    segments = preprocess_audio_to_segments(audio_path, segment_frames)

    all_probs = []
    all_preds = []

    for seg in segments:
        x = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs.squeeze(0).cpu())
            all_preds.append(torch.argmax(probs, dim=1).item())

    if VOTING == "majority":
        from collections import Counter
        final_pred = Counter(all_preds).most_common(1)[0][0]
        return final_pred, None, None  # 확률 없음
    elif VOTING == "probability":
        avg_prob = torch.stack(all_probs).mean(dim=0)
        final_pred = torch.argmax(avg_prob).item()
        pred_prob = avg_prob[final_pred].item()
        gt_prob = avg_prob[gt_target].item()
        return final_pred, pred_prob, gt_prob


# ----------------------------
# 전체 테스트 루프 (GT 기준 정렬 + 확률 출력)
# ----------------------------


if __name__ == "__main__":
    correct = 0
    total = 0

    if VOTING == "probability":
        print(f"{'File':30s} | {'GT (idx:name)':20s} | {'Pred (idx:name)':20s} | {'Pred%':7s} | {'GT%':7s} | Match")
        print("-" * 110)
    elif VOTING == "majority":
        print(f"{'File':30s} | {'GT (idx:name)':20s} | {'Pred (idx:name)':20s} | Match")
        print("-" * 90)

    # 파일 정렬
    files_with_labels = []
    for fname in os.listdir(AUDIO_DIR):
        if not fname.endswith(".wav"):
            continue
        try:
            target = int(fname.split('-')[3].split('.')[0])
            files_with_labels.append((target, fname))
        except:
            continue
    files_with_labels.sort(key=lambda x: x[0])

    for target, fname in files_with_labels:
        pred, pred_prob, gt_prob = predict(os.path.join(AUDIO_DIR, fname), target)

        gt_name = category_map.get(target, f"target{target}")
        pred_name = category_map.get(pred, f"target{pred}")
        match = "✔" if pred == target else "✘"

        gt_str = f"{target:02d}:{gt_name}"
        pred_str = f"{pred:02d}:{pred_name}"

        if VOTING == "probability":
            pred_prob_str = f"{pred_prob*100:6.2f}%" if pred_prob is not None else "  N/A"
            gt_prob_str = f"{gt_prob*100:6.2f}%" if gt_prob is not None else "  N/A"
            if target != pred:
                print(f"{fname:30s} | {gt_str:20s} | {pred_str:20s} | {pred_prob_str:>6s} | {gt_prob_str:>6s} |   {match}")
            else:
                print(f"{fname:30s} | {gt_str:20s} | {pred_str:20s} | {gt_prob_str:>6s} |    =    |   {match}")
        else:  # majority
            print(f"{fname:30s} | {gt_str:20s} | {pred_str:20s} |   {match}")

        total += 1
        correct += int(pred == target)

    acc = correct / total * 100 if total > 0 else 0

    if VOTING == "probability":
        print("-" * 110)
    else:
        print("-" * 90)
    print(f"[RESULT] Accuracy: {correct}/{total} ({acc:.2f}%)")

