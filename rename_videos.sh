#!/bin/bash

# 進入 online_exam1 目錄
cd /mnt/e/ResNet_video/online_test5 || exit

# 遍歷所有學號資料夾
for id in *; do
    if [ -d "$id/Camera_video" ]; then
        # 取得 Camera_video 裡的 mp4 檔案
        for file in "$id/Camera_video"/*.mp4; do
            if [ -f "$file" ]; then
                mv "$file" "$id/Camera_video/${id}_camera.mp4"
            fi
        done

        # 取得 Camera_video 裡的 csv 檔案
        for file in "$id/Camera_video"/*.csv; do
            if [ -f "$file" ]; then
                mv "$file" "$id/Camera_video/${id}_camera.csv"
            fi
        done
    fi
done
