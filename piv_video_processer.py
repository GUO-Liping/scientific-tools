# -*- coding: utf-8 -*-
# @Author: Liping Guo
# @Time: 2025/11/11
# @Function: extract frames from a (high-speed) video 

import cv2
import os
import concurrent.futures
import time


def save_frame(frame, frame_filename):
    """保存单帧图像"""
    cv2.imwrite(frame_filename, frame)


def extract_frames(video_path, output_folder, frame_interval, max_threads):
    """提取视频帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    frame_count = 0
    saved_count = 0
    futures = []
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
                futures.append(executor.submit(save_frame, frame, frame_filename))
                saved_count += 1

                # 减少频繁打印，仅每100帧输出一次
                if saved_count % 100 == 0:
                    print(f"已保存 {saved_count} 张帧图像...")

            frame_count += 1

        cap.release()

        # 等待所有线程完成
        for future in concurrent.futures.as_completed(futures):
            future.result()

    elapsed_time = time.time() - start_time
    return saved_count, frame_count, elapsed_time


# -------------------- 主程序部分 --------------------
video_path = '192.168.8.47-20241106-172713.avi'
#video_path = '192.168.8.22-20241106-172707.avi'
output_folder = 'frames_' + os.path.splitext(os.path.basename(video_path))[0]
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

# 若元数据中帧率异常（如 0 或 15），强制设定
fps = 1000.0
frame_interval = 10     # 每10帧提取一帧
time_interval = frame_interval / fps
duration = frame_count_total / fps
max_threads = 16
saved_count, frame_count, elapsed_time = extract_frames( video_path, output_folder, frame_interval, max_threads)

# -------------------- 视频信息 --------------------
print(f"\n原始视频帧数: {frame_count_total}")
print(f"原始视频帧率: {fps} fps")
print(f"原始视频时长: {duration:.2f} 秒 \n")

# -------------------- 输出结果 --------------------
print(f"提取完成！视频提取消耗时长: {elapsed_time:.2f} 秒")
print(f"提取完成！总计提取视频画面: {saved_count} 张")
print(f"提取完成！提取视频帧数间隔: {frame_interval} 帧")
print(f"提取完成！提取视频时间间隔: {time_interval:.4f} 秒")
