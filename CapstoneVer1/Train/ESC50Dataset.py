import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class ESC50Dataset(Dataset):
    def __init__(self, 
                 csv_file, 
                 npy_dir, 
                 fold,
                 segment_type='short',   # 'short' or 'long'
                 segment_frames=None,
                 overlap=None
                ):
        self.meta = pd.read_csv(csv_file)
        self.meta = self.meta[self.meta['fold'].isin(fold)]
        self.npy_dir = npy_dir

        # 자동 설정
        if segment_type == 'short':
            self.segment_frames = segment_frames or 41
            self.overlap = overlap if overlap is not None else 0.5
        elif segment_type == 'long':
            self.segment_frames = segment_frames or 101
            self.overlap = overlap if overlap is not None else 0.9
        else:
            raise ValueError("segment_type must be 'short' or 'long'")

        # 모든 segment 미리 생성
        self.segments = []
        self.labels = []

        print("[INFO] Preloading all segments...")
        for i in tqdm(range(len(self.meta))):
            row = self.meta.iloc[i]
            filename = row['filename'].replace('.wav', '.npy')
            label = row['target']
            mel_delta_path = os.path.join(self.npy_dir, filename)

            if not os.path.exists(mel_delta_path):
                print(f"[WARNING] Missing file: {mel_delta_path}")
                continue

            mel_delta = np.load(mel_delta_path)
            all_segments = self.segment_audio_features(mel_delta)

            self.segments.extend(all_segments)
            self.labels.extend([label] * len(all_segments))

        print(f"[INFO] Total segments loaded: {len(self.segments)}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x = self.segments[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), y

    def segment_audio_features(self, mel_delta):
        step = int(self.segment_frames * (1 - self.overlap))
        num_segments = (mel_delta.shape[2] - self.segment_frames) // step + 1
        segments = []

        for i in range(num_segments):
            start = i * step
            end = start + self.segment_frames
            seg = mel_delta[:, :, start:end]
            if seg.shape[2] == self.segment_frames:
                segments.append(seg)

        return segments
