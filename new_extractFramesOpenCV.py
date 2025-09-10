import os
import cv2
import argparse
import sys
import shutil
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from videos.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video files directory.')
    parser.add_argument('--destination', type=str, required=True, help='Destination directory for extracted frames.')
    args = parser.parse_args()

    videoPath = args.video_path
    videoPathFrames = args.destination

    # 與原行為一致：若輸出資料夾存在且非空，直接結束（視為已完成）
    if os.path.exists(videoPathFrames) and os.listdir(videoPathFrames):
        # 資料夾存在且裡面有東西
        sys.exit(0)

    os.makedirs(videoPathFrames, exist_ok=True)

    # ---------- 僅更換此段為 FFmpeg 實作，其他流程不變 ----------

    # 檢查 ffmpeg 是否可用
    if shutil.which("ffmpeg") is None:
        print("錯誤：系統未找到 ffmpeg，請先安裝 FFmpeg 後再執行。", file=sys.stderr)
        sys.exit(1)

    # 以 FFmpeg 抽幀：
    # -vsync 0：避免補幀/丟幀
    # -frame_pts 1：檔名以時間戳為基礎（若您更偏好嚴格 0,1,2… 連號，可移除此參數）
    # -start_number 0：從 0 開始命名，以貼近原程式輸出 0.jpg、1.jpg…
    # -q:v 3：JPEG 品質（2~5 常用；數字越小畫質越高、檔案越大），可視需求調整
    out_pattern = os.path.join(videoPathFrames, "%d.jpg")  # 對應 0,1,2… 連號
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", videoPath,
        "-vsync", "0",
        "-start_number", "0",
        "-q:v", "3",
        out_pattern
    ]

    # 如需以固定頻率（非逐幀）抽取，可改為加入： "-vf", "fps=1"
    # 例如：
    # cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i",videoPath,
    #        "-vf","fps=1","-vsync","0","-start_number","0","-q:v","3",out_pattern]

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print("FFmpeg 抽幀失敗，請確認輸入影片與 FFmpeg 環境。", file=sys.stderr)
            sys.exit(result.returncode)
    except Exception as e:
        print(f"執行 FFmpeg 時發生例外：{e}", file=sys.stderr)
        sys.exit(1)

    # ------------------------- 結束 FFmpeg 區塊 -------------------------

    # 與原版一致：程式結束前無其他動作
