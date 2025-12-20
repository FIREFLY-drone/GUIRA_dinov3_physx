import os
import glob
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

RAW_DATA_ROOT = "/workspaces/FIREPREVENTION/data/raw"
PROCESSED_DIR = "/workspaces/FIREPREVENTION/data/processed/smoke_timesformer"
MANIFEST_DIR = "/workspaces/FIREPREVENTION/data/manifests"
CLIP_LENGTH = 16
FPS = 8
STRIDE = 2
LABEL_THRESHOLD = 3  # At least 3 frames with smoke to label as smoke

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)

def extract_frames(video_path, out_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(out_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
    cap.release()
    return frames

def build_clips_from_frames(dataset_path, clip_length, step):
    """
    Builds clips from individual frames and their corresponding masks from the
    FLAME dataset's segmentation data structure.
    """
    clips = []
    labels = []

    frame_dir = os.path.join(dataset_path, 'frames', 'Segmentation', 'Data', 'Images')
    mask_dir = os.path.join(dataset_path, 'frames', 'Segmentation', 'Data', 'Masks')

    if not os.path.isdir(frame_dir) or not os.path.isdir(mask_dir) or not os.listdir(frame_dir):
        print(f"Warning: Frame or mask directory is missing or empty in {dataset_path}")
        print("Please ensure you have downloaded and extracted items 9 and 10 from the dataset's IEEE Dataport page")
        print("into '.../frames/Segmentation/Data/Images/' and '.../frames/Segmentation/Data/Masks/' respectively, as per the README.md.")
        return clips, labels

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])

    for i in range(0, len(frame_files) - clip_length + 1, step):
        clip_frames = []
        smoke_frame_count = 0

        for j in range(clip_length):
            frame_file = frame_files[i+j]
            frame_path = os.path.join(frame_dir, frame_file)
            # Masks are PNGs as per the dataset description
            mask_filename = os.path.splitext(frame_file)[0].replace('_rgb', '') + '.png'
            mask_path = os.path.join(mask_dir, mask_filename)

            if not os.path.exists(mask_path):
                continue

            clip_frames.append(frame_path)

            # Heuristic: check if mask file has substantial size, implying smoke is present.
            # A more robust method would be to load the mask and check pixel values.
            if os.path.getsize(mask_path) > 300: # Adjusted heuristic for non-empty mask
                smoke_frame_count += 1

        if len(clip_frames) == clip_length:
            clips.append(clip_frames)
            # Label clip as "smoke" if a minimum number of frames contain smoke
            labels.append(1 if smoke_frame_count >= LABEL_THRESHOLD else 0)

    return clips, labels

def convert_smoke_temporal():
    print("Processing smoke temporal datasets...")
    # Process the FLAME dataset (aliased as flame_rgb)
    flame_rgb_dir = os.path.join(RAW_DATA_ROOT, 'flame_rgb')
    clips, labels = build_clips_from_frames(flame_rgb_dir, clip_length=CLIP_LENGTH, step=STRIDE)

    # TODO: Add flame2_rgb_ir and wit_uas_thermal processing
    print(f"Extracted {len(clips)} clips from flame_rgb.")

    if not clips:
        print("No clips were extracted. Skipping train/test split and manifest generation.")
        return

    # Combine clips and labels for splitting
    clip_data = list(zip(clips, labels))

    # Check for sufficient data for stratified split
    if len(set(labels)) < 2:
        print("Warning: Only one class present in the data. Cannot perform stratified split. Using regular split.")
        train_val, test = train_test_split(clip_data, test_size=0.1, random_state=42)
        if train_val:
             train, val = train_test_split(train_val, test_size=0.2, random_state=42)
        else:
            train, val = [], []
    else:
        try:
            train_val, test = train_test_split(clip_data, test_size=0.1, random_state=42, stratify=[item[1] for item in clip_data])
            train, val = train_test_split(train_val, test_size=0.2, random_state=42, stratify=[item[1] for item in train_val])
        except ValueError:
            print("Warning: Not enough samples for stratified split. Falling back to regular split.")
            train_val, test = train_test_split(clip_data, test_size=0.1, random_state=42)
            train, val = train_test_split(train_val, test_size=0.2, random_state=42)


    # Save manifest
    for split, data in zip(['train', 'val', 'test'], [train, val, test]):
        if not data:
            print(f"Skipping manifest generation for '{split}' split as it is empty.")
            continue
        manifest_path = os.path.join(MANIFEST_DIR, f'smoke_timesformer_{split}.json')
        # Unzip the data for saving
        split_clips, split_labels = zip(*data)
        manifest_data = {
            "clips": split_clips,
            "labels": split_labels
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
    print("Saved manifests for train, val, test splits where data was available.")

if __name__ == '__main__':
    convert_smoke_temporal()
