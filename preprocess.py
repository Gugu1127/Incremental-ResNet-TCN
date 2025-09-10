import pandas as pd
import os
import subprocess
import shutil
import json
import glob
import argparse
import pathlib

# === 使用者影片根目錄 ===
video_folder_path = '/mnt/e/ResNet_video'

# === 輸出之訓練影像根目錄（依「JSON 檔名」分層）===
TRAIN_ROOT = './Train_images'

def read_json(path: str):
    # 讀取 json 的資料（維持原行為）
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)

    test_name = content.get("video_type")
    video_name = content.get("video_name")  # 例如 "11027261_....mp4"
    # 仍保留 std_id 以便尋找來源影片路徑
    std_id = video_name.split('_')[0] if video_name else None

    user_annots = content.get("User_annotation", [])
    tags = user_annots[0].get("tags", []) if user_annots else []

    return test_name, video_name, std_id, tags

def video2frame(video_path: str, destination: str):
    # 呼叫外部腳本擷取影格
    cmd = [
        'python3', 'new_extractFramesOpenCV.py',
        '--video_path', video_path,
        '--destination', destination
    ]
    subprocess.run(cmd, check=False)

def split_and_organize_images_by_tags(
    tag_list, source_folder, output_base_folder, std_id, fps=60, max_frames=280, total_frame_count=None
):
    """
    將標註區間切成固定長度之視窗（預設 280 張），輸出到 output_base_folder 下之多個資料夾，
    每個資料夾內有連續編號之 jpg（1.jpg...max_frames.jpg），不足者以末張補齊。
    """
    import os, glob, shutil

    # 僅這兩者視為「有標註」
    label_map = {
        "閱讀題目": 1,
        "思考解答": 2,
    }
    # 「視線校正」與未標註皆視為 0
    blank_cats = {"視線校正"}

    if total_frame_count is None:
        img_files = glob.glob(os.path.join(source_folder, "*.jpg"))
        total_frame_count = len(img_files) - 1 if len(img_files) > 0 else 0

    tag_list_sorted = sorted(tag_list, key=lambda x: float(x["start"]))

    intervals = []
    prev_end = 0
    for tag in tag_list_sorted:
        start = int(float(tag["start"]) * fps)
        end   = int(float(tag["end"])   * fps)
        cat   = tag["category"]

        if cat in blank_cats:
            intervals.append({"start": start, "end": end, "label": 0})
        else:
            label = label_map.get(cat, None)
            if label is None:
                intervals.append({"start": start, "end": end, "label": 0})
            else:
                if start > prev_end:
                    intervals.append({"start": prev_end, "end": start - 1, "label": 0})
                intervals.append({"start": start, "end": end, "label": label})
            prev_end = end + 1

    # 最尾端的空白
    if prev_end <= total_frame_count:
        intervals.append({"start": prev_end, "end": total_frame_count, "label": 0})

    # 合併連續的 label=0
    merged_intervals = []
    last = None
    for interval in sorted(intervals, key=lambda x: x["start"]):
        if interval["label"] == 0:
            if last and last["label"] == 0 and last["end"] + 1 >= interval["start"]:
                last["end"] = max(last["end"], interval["end"])
            else:
                merged_intervals.append(interval)
                last = merged_intervals[-1]
        else:
            merged_intervals.append(interval)
            last = merged_intervals[-1]

    folder_category_list = []
    for interval in merged_intervals:
        interval_start = interval["start"]
        interval_end   = interval["end"]
        label          = interval["label"]
        total_frames   = interval_end - interval_start + 1
        if total_frames <= 0:
            continue

        num_splits = (total_frames + max_frames - 1) // max_frames
        for split_idx in range(num_splits):
            split_start = interval_start + split_idx * max_frames
            split_end   = min(split_start + max_frames - 1, interval_end)
            folder_name = f"{std_id}_label{label}_{split_start}_{split_end}"

            # 關鍵：輸出至 Train_images/<run_key>/<folder_name>
            output_folder = os.path.join(output_base_folder, folder_name)
            os.makedirs(output_folder, exist_ok=True)

            copied_imgs = []
            for img_idx, frame_num in enumerate(range(split_start, split_end + 1)):
                img_name = f"{frame_num}.jpg"
                src_img_path = os.path.join(source_folder, img_name)
                dst_img_path = os.path.join(output_folder, f"{img_idx + 1}.jpg")
                if os.path.exists(src_img_path):
                    shutil.copy(src_img_path, dst_img_path)
                    copied_imgs.append(dst_img_path)
                else:
                    print(f"警告：找不到圖片 {src_img_path}")

            # 若不足 max_frames，以末張補齊
            actual_frames = len(copied_imgs)
            if actual_frames < max_frames:
                if actual_frames > 0:
                    last_img = copied_imgs[-1]
                    for extra_idx in range(actual_frames + 1, max_frames + 1):
                        dst_img_path = os.path.join(output_folder, f"{extra_idx}.jpg")
                        shutil.copy(last_img, dst_img_path)
                else:
                    print(f"錯誤：{folder_name} 該分段沒有圖片！")

            folder_category_list.append({
                "folder_name": folder_name,  # 相對於 data_root（Train_images/<run_key>）
                "label": label,
            })

    return folder_category_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video annotations and extract frames.')
    parser.add_argument('--file-path', type=str, required=True, help='Path to a single annotation JSON.')
    args = parser.parse_args()

    # 允許上層傳入帶引號的路徑
    json_file = args.file_path.strip('"').strip("'")
    run_key = pathlib.Path(json_file).stem  # 以 JSON 檔名（去 .json）作為資料根目錄名
    print(f"[preprocess] run_key = {run_key}", flush=True)

    test_name, video_name, std_id, tags = read_json(json_file)
    print(f"Processing {video_name} for student {std_id} in test {test_name}", flush=True)

    # ---- 影格輸出（維持到受試者目錄底下的 Camera_videoFrames）----
    std_video_path      = os.path.join(video_folder_path, test_name, std_id)
    camera_video_path   = os.path.join(std_video_path, 'Camera_video')
    video_files = [f for f in os.listdir(camera_video_path) if f.endswith('.mp4')]
    assert len(video_files) > 0, f"No .mp4 found under {camera_video_path}"
    video_path = os.path.join(camera_video_path, video_files[0])
    frames_dir = os.path.join(std_video_path, 'Camera_videoFrames')
    video2frame(video_path, frames_dir)

    # ---- 關鍵變更：切片輸出到 Train_images/<run_key>/ 並在每次執行前清空該目錄----
    run_train_root = os.path.join(TRAIN_ROOT, run_key)
    os.makedirs(run_train_root, exist_ok=True)

    info_list = split_and_organize_images_by_tags(
        tag_list=tags,
        source_folder=frames_dir,
        output_base_folder=run_train_root,  # <- Train_images/<run_key>
        std_id=std_id
    )

    # 僅用本次影片資料重建 train.csv（確保單片訓練）
    rows = [{"path": info["folder_name"], "label": info["label"]} for info in info_list]
    df = pd.DataFrame(rows)
    csv_out = os.path.join(run_train_root, 'train.csv')
    df.to_csv(csv_out, index=False)
    print(f"✅ 已輸出切片與 CSV：{csv_out}", flush=True)
