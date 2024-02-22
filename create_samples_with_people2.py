import os
import cv2
import numpy as np
import logging
import time
import argparse
import glob
from ultralytics import YOLO
from typing import List
import memory_profiler


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

    video_extractor = VideoExtractor(cam1_path, cam2_path)
    img_buffer = video_extractor.get_sequence(args.gap, args.start_frame, args.frame_limit)
    print(len(img_buffer))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', type=str, default='/mnt/data/reid_datasets/Pair-1')#r'D:\datasets\reid\polytech\Pair-1')
    parser.add_argument('--dst', type=str, default='/mnt/data/reid_datasets/Pair-1')#r'D:\datasets\reid\polytech\Pair-1')
    parser.add_argument('--gap', type=int, default=100)
    parser.add_argument('--frame_limit', type=int, default=2000)
    parser.add_argument('--start_frame', type=int, default=1100)

    args = parser.parse_args()
    return args


class VideoExtractor:
    def __init__(self, cam1_path: str, cam2_path: str) -> None:
        self.cam1_path = cam1_path
        self.cam2_path = cam2_path

        self.video1 = cv2.VideoCapture(cam1_path)
        self.video2 = cv2.VideoCapture(cam2_path)

        self.fps1 = self.video1.get(cv2.CAP_PROP_FPS)
        self.fps2 = self.video2.get(cv2.CAP_PROP_FPS)

        self.main_fps = min(self.fps1, self.fps2)

        self.frame_count_1 = 0  # Counter of frames in video 1
        self.frame_count_2 = 0  # Counter of frames in video 2
        self.main_frame_count = 0

        self.model = YOLO('yolov8s.pt')

        print('fps1:', self.fps1)
        print('fps2:', self.fps2)

        self.frame_idx_buffer = []
        

    def get_sequence(self, gap: int, start_frame: int, frame_limit: int):
        last_seen_people = self.main_frame_count
        seen_people_at_least_once = False

        if self.main_frame_count >= frame_limit:
            return None

        while True:
            print(self.main_frame_count, self.frame_count_1, self.frame_count_2)
            ret, frame1, frame2 = self.read_frames()

            if self.main_frame_count < start_frame:
                last_seen_people = self.main_frame_count
                continue

            if ret is False:
                return self.frame_idx_buffer
            
            self.frame_idx_buffer.append(self.main_frame_count)

            if seen_people_at_least_once and self.main_frame_count - last_seen_people > gap:
                # img_buffer[max(0, len(img_buffer) - gap):]
                return self.frame_idx_buffer
            
            if self.main_frame_count - last_seen_people > gap:
                self.frame_idx_buffer.pop(0)
            
            if self.main_frame_count > frame_limit:
                return self.frame_idx_buffer
            
            has_people = check_if_has_people(self.model, [frame1, frame2])
            if has_people:
                last_seen_people = self.main_frame_count
                seen_people_at_least_once = True
                print('people')
        
        return img_buffer
            
    def read_frames(self):
        ret1, ret2 = True, True
        
        # FPS gains (show how much main fps is bigger than the others)
        fps_gain1 = self.main_fps / self.fps1
        fps_gain2 = self.main_fps / self.fps2
        
        while self.frame_count_1 <= self.main_frame_count:
            ret1, frame1 = self.video1.read()
            self.frame_count_1 += fps_gain1
        
        while self.frame_count_2 <= self.main_frame_count:
            ret2, frame2 = self.video2.read()
            self.frame_count_2 += fps_gain2
            
        self.main_frame_count += 1
        common_ret = ret1 and ret2
        return common_ret, frame1, frame2


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
