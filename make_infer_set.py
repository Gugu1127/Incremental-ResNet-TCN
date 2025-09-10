import os
import argparse
import pandas as pd
import shutil
import glob
import pathlib

video_folder_path = '/mnt/e/ResNet_video'
INFER_ROOT = './Infer_images'

def split_frames_to_infer_set(frames_dir, output_base_folder, std_id, max_frames=280):
    img_files = glob.glob(os.path.join(frames_dir, "*.jpg"))
    img_files = sorted(img_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
    total_frames = len(img_files)
    if total_frames == 0:
        raise RuntimeError(f"找不到影格: {frames_dir}")

    rows = []
    num_splits = (total_frames + max_frames - 1) // max_frames
    for split_idx in range(num_splits):
        split_start = split_idx * max_frames
        split_end = min(split_start + max_frames, total_frames)

        folder_name = f"{std_id}_label4_{split_start}_{split_end-1}"
        output_folder = os.path.join(output_base_folder, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        copied_imgs = []
        for i, src in enumerate(img_files[split_start:split_end], start=1):
            dst = os.path.join(output_folder, f"{i}.jpg")
            shutil.copy(src, dst)
            copied_imgs.append(dst)

        # 若不足 max_frames，以末張補齊
        actual_frames = len(copied_imgs)
        if actual_frames < max_frames and actual_frames > 0:
            last_img = copied_imgs[-1]
            for extra_idx in range(actual_frames+1, max_frames+1):
                dst = os.path.join(output_folder, f"{extra_idx}.jpg")
                shutil.copy(last_img, dst)

        rows.append({"path": folder_name, "label": 4})

    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build inference dataset (no annotation, label=4).")
    parser.add_argument("--file-path", type=str, required=True, help="Path to annotation JSON, 用於定位 test_name/std_id")
    args = parser.parse_args()

    json_file = args.file_path.strip('"').strip("'")
    run_key = pathlib.Path(json_file).stem
    print(f"[infer-prep] run_key = {run_key}")

    import json
    with open(json_file, "r", encoding="utf-8") as f:
        content = json.load(f)
    test_name = content.get("video_type")
    video_name = content.get("video_name")
    std_id = video_name.split('_')[0] if video_name else None

    frames_dir = os.path.join(video_folder_path, test_name, std_id, 'Camera_videoFrames')
    if not os.path.isdir(frames_dir):
        raise RuntimeError(f"影格目錄不存在: {frames_dir}，請先執行 preprocess/video2frame")

    run_infer_root = os.path.join(INFER_ROOT, run_key)
    os.makedirs(run_infer_root, exist_ok=True)

    rows = split_frames_to_infer_set(frames_dir, run_infer_root, std_id)
    df = pd.DataFrame(rows)
    csv_out = os.path.join(run_infer_root, "infer.csv")
    df.to_csv(csv_out, index=False)
    print(f"✅ 已建立推理集: {csv_out}")
