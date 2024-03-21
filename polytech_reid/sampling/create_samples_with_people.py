import os
import cv2
import numpy as np
import logging
import time
import argparse
import glob
from ultralytics import YOLO
from typing import List

logging.getLogger('ultralytics').handlers = [logging.NullHandler()]


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.dst, '1'), exist_ok=True)
    os.makedirs(os.path.join(args.dst, '2.1'), exist_ok=True)
    os.makedirs(os.path.join(args.dst, '2.2'), exist_ok=True)
    
    cam1_paths = glob.glob(os.path.join(args.src, '*Cam1*.mp4'))
    cam2_paths = glob.glob(os.path.join(args.src, '*Cam2*.mp4'))

    if len(cam1_paths) == 0 or len(cam2_paths) == 0:
        print('Not enough data')
        return

    cam1_path = cam1_paths[0]
    cam2_path = cam2_paths[0]

    cam1_name = os.path.splitext(os.path.basename(cam1_path))[0]
    cam2_name = os.path.splitext(os.path.basename(cam2_path))[0]

    video1 = cv2.VideoCapture(cam1_path)
    video2 = cv2.VideoCapture(cam2_path)

    fps1 = video1.get(cv2.CAP_PROP_FPS)
    fps2 = video2.get(cv2.CAP_PROP_FPS)
    
    main_fps = fps2
    fps_gain = fps1 / fps2
    
    print('cam1_path:', cam1_path)
    print('cam2_path:', cam2_path)

    print('fps1:', video1.get(cv2.CAP_PROP_FPS))
    print('fps2:', video2.get(cv2.CAP_PROP_FPS))

    print('frame_count_1:', video1.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame_count_2:', video2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    model = YOLO('yolov8s.pt')
    frame_count = 0
    cnt1 = 0
    cnt2 = 0

    img_buffer = []
    prev_has_people = False
    lost_people = False

    while True:
        print(frame_count, cnt1, cnt2)
        ret, frame1, frame2, cnt1, cnt2, frame_count = read_frames(video1, video2, frame_count, cnt1, cnt2, fps_gain)

        if ret is False:
            continue
        
        img_buffer.append([frame1, frame2])
        if len(img_buffer) > args.gap:
            if lost_people:
                lost_people = False
                for i, imgs in enumerate(img_buffer):
                    save_frames(args.dst, frame_count - len(img_buffer) + 1 + i, imgs[0], imgs[1])
            
            img_buffer.pop(0)
        
        has_people = check_if_has_people(model, [frame1, frame2])
        if has_people:
            print('people', frame_count)
            for i, imgs in enumerate(img_buffer):
                save_frames(args.dst, frame_count - len(img_buffer) + 1 + i, imgs[0], imgs[1])
            img_buffer = []
        
        if prev_has_people is True and has_people is False:
            lost_people = True

        prev_has_people = has_people

        if frame_count > args.frame_limit:
            break
        

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', type=str, default=r'D:\datasets\reid\polytech\Pair-1')
    parser.add_argument('--dst', type=str, default=r'D:\datasets\reid\polytech\Pair-1')
    parser.add_argument('--gap', type=int, default=100)
    parser.add_argument('--frame_limit', type=int, default=1000)

    args = parser.parse_args()
    return args


def read_frames(video1: cv2.VideoCapture, video2: cv2.VideoCapture, frame_count: int, cnt1: int, cnt2: int, fps_gain: float):
    ret1, ret2 = True, True
    
    while cnt1 <= frame_count:
        ret1, frame1 = video1.read()
        cnt1 += 1
    
    while cnt2 <= frame_count:
        ret2, frame2 = video2.read()
        cnt2 += fps_gain
        
    frame_count += 1
    return ret1 and ret2, frame1, frame2, cnt1, cnt2, frame_count


def save_frames(dst: str, frame_count: int, frame1: np.ndarray, frame2: np.ndarray):
    cv2.imwrite(os.path.join(dst, '1', f"{frame_count}.jpg"), frame1)
    cv2.imwrite(os.path.join(dst, '2.1', f"{frame_count}.jpg"), frame2[:len(frame2) // 2])
    cv2.imwrite(os.path.join(dst, '2.2', f"{frame_count}.jpg"), frame2[len(frame2) // 2:])


def check_if_has_people(model: YOLO, frames: List[np.ndarray]) -> bool:
    results = model(frames, stream=True)
    for result in results:
        for c in result.boxes.cls:
            if model.names[int(c.item())] == 'person':
                return True
    return False


def create_samples_1(video: cv2.VideoCapture, name: str, fps_gain: float, dst_dir: str, frame_limit: int):
    os.makedirs(os.path.join(dst_dir, name), exist_ok=True)
    cnt = 0
    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        if cnt >= frame_limit and frame_limit != -1:
            break
        
        ret, frame = video.read()
        if ret is False:
            break

        if i * fps_gain >= cnt:
            frame_path = os.path.join(dst_dir, name, f"{name}_{cnt}.jpg")
            cv2.imwrite(frame_path, frame)
            cnt += 1
            print(cnt, frame_path)
            
    video.release()
    

def create_samples_2(video: cv2.VideoCapture, name: str, dst_dir: str, frame_limit: int):
    os.makedirs(os.path.join(dst_dir, f"{name}.1"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, f"{name}.2"), exist_ok=True)
    
    for cnt in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        if cnt >= frame_limit and frame_limit != -1:
            break
        
        ret, frame = video.read()
        if ret is False:
            break
        
        for i in range(2):
            
            start_idx = len(frame) // 2 * i
            stop_idx = len(frame) // 2 * (i + 1)
            subframe = frame[start_idx: stop_idx]
            
            subframe_path = os.path.join(dst_dir, f"{name}.{i + 1}", f"{name}.{i + 1}_{cnt}.jpg")
            cv2.imwrite(subframe_path, subframe)
            print(cnt, subframe_path)
        
    video.release()


if __name__ == '__main__' :
    main()
