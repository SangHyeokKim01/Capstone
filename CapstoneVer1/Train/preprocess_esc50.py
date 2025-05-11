import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
"""
음성데이터를 mel spectrogram으로 바꿔준다.
안바꾸고 러닝돌리면 epoch마다 바꿔주는 계산을 해야해서 번거로움
"""
csv_path = "./ESC-50/meta/esc50.csv"
audio_path = "./ESC-50/audio"
save_dir = "./preprocessed_mel"

sr = 22050
n_mels = 60

os.makedirs(save_dir, exist_ok=True)

meta = pd.read_csv(csv_path)

for filename in tqdm(meta['filename']):
    y, _ = librosa.load(os.path.join(audio_path, filename), sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(S=mel)
    delta = librosa.feature.delta(data=log_mel)
    mel_delta = np.stack([log_mel, delta], axis=0)  # shape: (2, n_mels, T)

    np.save(os.path.join(save_dir, filename.replace(".wav", ".npy")), mel_delta)
