import os
import json
import glob
import shutil
import argparse
import subprocess
import pandas as pd

# 根目錄：實際影片與個人資料夾所在處（沿用您原本的設定）
VIDEO_ROOT = '/mnt/e/ResNet_video'

# 視窗大小（每一個訓練片段的影格數）
MAX_FRAMES = 280
# 以何種 FPS 的時間標註換算為影格編號（依您先前版本使用 60）
FPS = 60

def read_json(path: str):
    """讀取 JSON，回傳 (test_name, std_id, tag_list)"""
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)

    test_name = content.get("Test_name") or content.get("test_name")
    std_id = content.get("std_id") or content.get("Std_ID") or content.get("StdId") or content.get("student_id")

    # 標註欄位名稱在不同來源可能不同，這裡盡量兼容
    user_annots = content.get("User_annotation") or content.get("UserAnnotation") or content.get("annotation") or []
    if not isinstance(user_annots, list):
        user_annots = []

    # 正規化為 [{'start': float, 'end': float, 'label': int}, ...]
    tags = []
    for t in user_annots:
        start = float(t.get("start", 0.0))
        end = float(t.get("end", 0.0))
        # 您先前版只將兩類視為「有標註」：1/2，其餘視為 0（空白）
        # 若 JSON 以字串記，轉成 int；若缺漏則 0
        raw_label = t.get("type") or t.get("label") or 0
        try:
            lab = int(raw_label)
        except Exception:
            lab = 0
        lab = lab if lab in (1, 2) else 0
        if end < start:  # 修正意外資料
            start, end = end, start
        tags.append({"start": start, "end": end, "label": lab})

    if not test_name or not std_id:
        raise ValueError("JSON 內缺少 test_name/std_id，請確認欄位。")

    return test_name, std_id, tags


def run_video2frame(video_path: str, out_dir: str):
    """呼叫既有的 new_extractFramesOpenCV.py 將影片轉為影格。"""
    os.makedirs(out_dir, exist_ok=True)
    # 假設您的外部腳本使用參數順序：<video_path> <save_dir>
    cmd = ['python3', 'new_extractFramesOpenCV.py', video_path, out_dir]
    print(f"👉 轉檔：{' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def build_intervals(tag_list, total_frame_count=None):
    """由標註建立含空白區間的整體區段列表（以影格為單位）。"""
    # 轉成影格座標
    slots = []
    for t in tag_list:
        s = int(round(float(t["start"]) * FPS))
        e = int(round(float(t["end"]) * FPS))
        lab = int(t["label"])
        slots.append((s, e, lab))

    # 依起點排序
    slots.sort(key=lambda x: (x[0], x[1]))

    intervals = []
    prev_end = 0
    for (s, e, lab) in slots:
        if s > prev_end + 1:
            # 空白區間（標記為 0）
            intervals.append({"start": prev_end + 1, "end": s - 1, "label": 0})
        intervals.append({"start": s, "end": e, "label": lab})
        prev_end = max(prev_end, e)

    # 若有總影格數，可補尾端空白
    if total_frame_count is not None and prev_end < total_frame_count:
        intervals.append({"start": prev_end + 1, "end": total_frame_count, "label": 0})

    # 合併相鄰且標籤相同的區段
    merged = []
    for iv in intervals:
        if not merged:
            merged.append(iv)
        else:
            last = merged[-1]
            if last["label"] == iv["label"] and last["end"] + 1 >= iv["start"]:
                last["end"] = max(last["end"], iv["end"])
            else:
                merged.append(iv)
    return merged


def split_copy_frames_by_intervals(source_folder: str, output_base_folder: str,
                                   std_id: str, intervals, max_frames=MAX_FRAMES):
    """
    依區段切 MAX_FRAMES 的片段，複製影格至對應資料夾；若不足則以末張補齊到 MAX_FRAMES。
    回傳：list[{'folder_rel': <相對CWD路徑>, 'label': int}]
    """
    os.makedirs(output_base_folder, exist_ok=True)
    info_list = []

    for interval in intervals:
        s = interval["start"]
        e = interval["end"]
        lab = interval["label"]
        total = e - s + 1
        if total <= 0:
            continue

        num_splits = (total + max_frames - 1) // max_frames
        for k in range(num_splits):
            split_start = s + k * max_frames
            split_end = min(split_start + max_frames - 1, e)
            folder_name = f"{std_id}_label{lab}_{split_start}_{split_end}"
            out_dir = os.path.join(output_base_folder, folder_name)
            os.makedirs(out_dir, exist_ok=True)

            copied = []
            for i, frame_num in enumerate(range(split_start, split_end + 1), start=1):
                src = os.path.join(source_folder, f"{frame_num}.jpg")
                dst = os.path.join(out_dir, f"{i}.jpg")
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    copied.append(dst)
                else:
                    print(f"⚠️ 找不到影格：{src}")

            # 影格不足則重複最後一張補齊到 MAX_FRAMES
            if len(copied) < max_frames and len(copied) > 0:
                last_img = copied[-1]
                for extra_idx in range(len(copied) + 1, max_frames + 1):
                    dst = os.path.join(out_dir, f"{extra_idx}.jpg")
                    shutil.copy(last_img, dst)

            # 記錄相對 CWD 的路徑
            rel_out_dir = os.path.relpath(out_dir, start=os.getcwd())
            info_list.append({"folder_rel": rel_out_dir, "label": lab})

    return info_list


def main():
    parser = argparse.ArgumentParser(description="Preprocess a single JSON → frames and CSV (per person).")
    parser.add_argument("--file-path", type=str, required=True, help="標註 JSON 檔（含路徑）")
    args = parser.parse_args()

    json_path = args.file_path.strip('"').strip("'")
    test_name, std_id, tags = read_json(json_path)

    # 個人資料夾
    std_dir = os.path.join(VIDEO_ROOT, test_name, std_id)
    cam_video_dir = os.path.join(std_dir, "Camera_video")
    frames_dir = os.path.join(std_dir, "Camera_videoFrames")  # 仿照 video2frame 的輸出位置
    per_person_train_root = os.path.join(std_dir, "images_train")  # ★ 新增：每個人自己的訓練資料夾

    # 找到一支 MP4（若有多支，取第一支）
    mp4s = sorted([f for f in os.listdir(cam_video_dir) if f.lower().endswith(".mp4")])
    if not mp4s:
        raise FileNotFoundError(f"在 {cam_video_dir} 找不到 MP4")
    video_path = os.path.join(cam_video_dir, mp4s[0])

    # 1) 影片 → 影格
    run_video2frame(video_path, frames_dir)

    # 若您有「總影格數」可由 frames_dir 推得
    jpgs = glob.glob(os.path.join(frames_dir, "*.jpg"))
    total_frames = len(jpgs) if jpgs else None

    # 2) 標註 → 區段（含空白）
    intervals = build_intervals(tags, total_frame_count=total_frames)

    # 3) 依區段切片並複製到「個人資料夾」底下的 images_train
    info = split_copy_frames_by_intervals(
        source_folder=frames_dir,
        output_base_folder=per_person_train_root,
        std_id=std_id,
        intervals=intervals,
        max_frames=MAX_FRAMES,
    )

    # 4) 產生 train.csv（單次只針對本 JSON；相對路徑便於訓練端載入）
    rows = [{"path": e["folder_rel"], "label": e["label"]} for e in info]
    df = pd.DataFrame(rows)
    df.to_csv("train.csv", index=False, encoding="utf-8")
    print(f"✅ 已產生 train.csv（{len(df)} 筆），並將資料存放於：{per_person_train_root}", flush=True)


if __name__ == "__main__":
    main()
