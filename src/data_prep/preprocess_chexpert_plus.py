import pandas as pd
import os
import numpy as np

DATA_DIR = os.environ.get('CHEXPERT_PLUS_DIR', './data/chexpert_plus')
IMG_DIR = os.path.join(DATA_DIR, 'png_valid')

# Load CSV
df = pd.read_csv(
    os.path.join(DATA_DIR, 'df_chexpert_plus_240401.csv'),
    on_bad_lines='skip',
    engine='python'
)

# Filter for validation set with findings
df = df[df['path_to_image'].str.contains('valid', na=False)]
df = df[df['section_findings'].apply(lambda x: isinstance(x, str) and len(x.split()) >= 2)]
print(f'Validation with findings: {len(df)}')

# Build image paths
def build_path(row):
    orig = str(row['path_to_image'])
    # valid/patient64542/study1/view1_frontal.jpg
    # → png_valid/patient64542/study1/view1_frontal.png
    stripped = orig.replace('valid/', '')
    png_name = stripped.replace('.jpg', '.png')
    return os.path.join(IMG_DIR, png_name)

df['image_path'] = df.apply(build_path, axis=1)
existing = df['image_path'].apply(os.path.exists).sum()
print(f'Images found: {existing}/{len(df)}')

# Show samples
for i in range(min(3, len(df))):
    path = df['image_path'].iloc[i]
    exists = os.path.exists(path)
    print(f'  [{i}] {path} (exists: {exists})')
    print(f'       Finding: {df["section_findings"].iloc[i][:100]}...')

# Only keep samples where image exists
df = df[df['image_path'].apply(os.path.exists)]
print(f'\nFinal samples with images: {len(df)}')

if len(df) > 0:
    df = df.rename(columns={'section_findings': 'findings'})
    output = './data/processed/chexpert_plus_test_clean.csv'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df[['image_path', 'findings']].to_csv(output, index=False)
    
    avg_len = df['findings'].astype(str).apply(lambda x: len(x.split())).mean()
    print(f'Saved: {output}')
    print(f'Avg findings length: {avg_len:.1f} words')
else:
    print('ERROR: No matching images found!')
    print('Check image directory structure:')
    for root, dirs, files in os.walk(IMG_DIR):
        for f in files[:3]:
            print(f'  {os.path.join(root, f)}')
        if files:
            break
