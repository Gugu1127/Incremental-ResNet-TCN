import os
import glob
import shutil
import json
import argparse
import pandas as pd

# 與原始程式一致的影片根路徑
video_folder_path = '/mnt/e/ResNet_video'

def read_json(path: str):
    """
    與原始程式相容的 JSON 解析：
    - 取出 test_name(video_type), video_name
    - 從 video_name 推出 std_id（video_name 以 stdId_ 開頭）
    """
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)

    test_name = content.get("video_type")
    video_name = content.get("video_name")
    std_id = video_name.split('_')[0] if video_name else None
    return test_name, video_name, std_id


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def natural_key_from_filename(fp: str):
    """
    嘗試以檔名（不含副檔名）的整數值排序；若失敗則退回字串排序。
    例：'0.jpg'、'1.jpg'、'12.jpg' 會被轉為 0,1,12 排序。
    """
    base = os.path.basename(fp)
    name, _ = os.path.splitext(base)
    try:
        return int(name)
    except ValueError:
        return name


def split_frames_as_test(
    frames_dir: str,
    output_base: str,
    std_id: str,
    max_frames: int = 280
):
    """
    將 frames_dir 內的所有 *.jpg 以 max_frames 分段輸出到 output_base。
    - 子資料夾命名：{std_id}_label4_{startIdx}_{endIdx}
      其中 startIdx / endIdx 以原始影格檔名（數字）為準
    - 不足 max_frames 以該段「最後一張」影格複製填補
    - 回傳供 CSV 使用的清單：[{folder_name: ..., label: 4}, ...]
    """
    ensure_dir(output_base)

    img_files = glob.glob(os.path.join(frames_dir, "*.jpg"))
    if not img_files:
        raise FileNotFoundError(f"找不到影格：{frames_dir}/*.jpg")

    # 依影格編號自然排序
    img_files.sort(key=natural_key_from_filename)

    # 將影格的「原始編號」取出，以利命名
    frame_numbers = []
    for fp in img_files:
        name = os.path.splitext(os.path.basename(fp))[0]
        try:
            frame_numbers.append(int(name))
        except ValueError:
            # 若檔名非純數字，仍保留；以索引替代
            frame_numbers.append(None)

    # 逐段切分
    csv_rows = []
    n = len(img_files)
    idx = 0
    seg_idx = 0
    while idx < n:
        seg_files = img_files[idx: idx + max_frames]
        # 取得該段的 start/end（以原檔名中的數字為優先，否則以序號代替）
        start_num = frame_numbers[idx] if frame_numbers[idx] is not None else idx
        end_raw_idx = idx + len(seg_files) - 1
        end_num = frame_numbers[end_raw_idx] if frame_numbers[end_raw_idx] is not None else end_raw_idx

        folder_name = f"{std_id}_label4_{start_num}_{end_num}"
        out_dir = os.path.join(output_base, folder_name)
        ensure_dir(out_dir)

        # 複製並以 1.jpg ~ k.jpg 連續命名
        copied = []
        for i, src in enumerate(seg_files, start=1):
            dst = os.path.join(out_dir, f"{i}.jpg")
            shutil.copy(src, dst)
            copied.append(dst)

        # 若不足 max_frames，以該段最後一張影格複製填補至 max_frames
        if len(seg_files) < max_frames:
            if not copied:
                # 理論上不會發生，保險起見
                raise RuntimeError(f"{folder_name} 無影格可填補。")
            last_img = copied[-1]
            for i in range(len(seg_files) + 1, max_frames + 1):
                dst = os.path.join(out_dir, f"{i}.jpg")
                shutil.copy(last_img, dst)

        csv_rows.append({"path": folder_name, "label": 4})
        idx += max_frames
        seg_idx += 1

    return csv_rows


def main():
    parser = argparse.ArgumentParser(description="Build test set from frames with max_frames=280 and label=4.")
    parser.add_argument('--file-path', type=str, required=True,
                        help='標註 JSON 檔路徑（用於解析 test_name / std_id 與定位 Camera_videoFrames）')
    parser.add_argument('--max-frames', type=int, default=280, help='每段影格數（預設 280）')
    parser.add_argument('--output-dir', type=str, default='./images_test', help='測試集影像輸出根目錄（預設 ./images_test）')
    parser.add_argument('--csv-path', type=str, default='test.csv', help='輸出 CSV 路徑（預設 test.csv）')
    args = parser.parse_args()

    test_name, video_name, std_id = read_json(args.file_path)
    if not (test_name and std_id):
        raise ValueError("無法從 JSON 解析出 test_name 或 std_id，請確認 video_type / video_name 格式。")

    # 依原設計路徑定位 frames 位置
    std_video_path = os.path.join(video_folder_path, test_name, std_id)
    frames_dir = os.path.join(std_video_path, 'Camera_videoFrames')
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"找不到影格資料夾：{frames_dir}（請先完成影片轉影格步驟）")

    print(f"建立測試集：std_id={std_id}, test_name={test_name}")
    print(f"來源影格資料夾：{frames_dir}")
    print(f"輸出影像根目錄：{args.output_dir}")
    print(f"輸出 CSV：{args.csv_path}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 切段並輸出資料夾
    rows = split_frames_as_test(
        frames_dir=frames_dir,
        output_base=args.output_dir,
        std_id=std_id,
        max_frames=args.max_frames
    )

    # 產生/覆寫 test.csv（全部標籤=4）
    df = pd.DataFrame(rows)  # 欄位：path, label
    df.to_csv(args.csv_path, index=False)
    print(f"完成，共輸出 {len(rows)} 段。")


if __name__ == "__main__":
    main()
