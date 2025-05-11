import pandas as pd
import os

# [1] 메타 파일에서 target → category 자동 매핑
meta_path = '../ESC-50/meta/esc50.csv'
meta_df = pd.read_csv(meta_path)
category_map = meta_df.drop_duplicates('target')[['target', 'category']] \
                      .set_index('target')['category'].to_dict()

# target → category 딕셔너리 (번호순 정렬)
sorted_category_map = dict(sorted(category_map.items()))

# 보기 좋게 출력
print("ESC-50 Category Map (Sorted by Target Index):")
for target, name in sorted_category_map.items():
    print(f"{target:2d}: {name}")

# [2] .wav 파일이 들어 있는 폴더 경로
data_dir = './TestAudio'  # 분석할 오디오 파일 폴더
file_list = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

# [3] 메타 정보 생성
data = []
for fname in file_list:
    parts = fname.split('-')  # ['fold', 'src_file', 'A', 'target.wav']
    if len(parts) != 4 or not parts[3].endswith('.wav'):
        continue

    try:
        fold = int(parts[0])
        src_file = int(parts[1])
        target = int(parts[3].split('.')[0])
        category = category_map.get(target, '')  # 자동 매핑
    except ValueError:
        continue

    data.append({
        'filename': fname,
        'fold': fold,
        'target': target,
        'category': category,
        'esc10': False,
        'src_file': src_file
    })

# [4] 저장
df = pd.DataFrame(data)
df.to_csv('./meta/esc50_Test.csv', index=False)
print(df.head())
