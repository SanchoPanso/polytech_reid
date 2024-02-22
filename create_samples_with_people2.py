import os
import cv2
import numpy as np
import logging
import time
import argparse
import glob
from ultralytics import YOLO
from typing import List, Callable
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

    video_extractor = VideoExtractor([cam1_path, cam2_path])
    seq = video_extractor.get_sequence(args.gap, args.start_frame, args.frame_limit)
    print(seq)
    video_extractor.save_sequence(seq[0], seq[1], 'test_seq', save_frames)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', type=str, default=r'D:\datasets\reid\polytech\Pair-1')
    parser.add_argument('--dst', type=str, default=r'D:\datasets\reid\polytech\Pair-1')
    parser.add_argument('--gap', type=int, default=100)
    parser.add_argument('--frame_limit', type=int, default=5000)
    parser.add_argument('--start_frame', type=int, default=1000)

    args = parser.parse_args()
    return args


class VideoExtractor:
    def __init__(self, cam_paths: List[str]) -> None:
        self.cam_paths = cam_paths
        self.videos = [cv2.VideoCapture(path) for path in self.cam_paths]
        self.fps_list = [video.get(cv2.CAP_PROP_FPS) for video in self.videos]
        
        self.main_fps = min(self.fps_list)
        self.model = YOLO('yolov8s.pt')

        self.frame_counts = [0 for v in self.videos]
        self.main_frame_count = 0

    def reset(self):
        self.set_position(self, 0)


    def get_sequence(self, gap: int, start_frame: int, stop_frame: int, last_seen_people=None):
        
        self.set_position(start_frame)
        
        start_seq_idx = self.main_frame_count
        end_seq_idx = self.main_frame_count
        last_seen_people = last_seen_people or self.main_frame_count
        seen_people_at_least_once = False

        if self.main_frame_count >= stop_frame:
            return start_seq_idx, end_seq_idx

        while True:
            print(self.main_frame_count, *self.frame_counts)
            ret, frames = self.read_frames()

            if ret is False:
                return start_seq_idx, end_seq_idx
            
            end_seq_idx = self.main_frame_count

            if seen_people_at_least_once and self.main_frame_count - last_seen_people > gap:
                return start_seq_idx, end_seq_idx
            
            if self.main_frame_count - last_seen_people > gap:
                start_seq_idx = self.main_frame_count - gap
            
            if self.main_frame_count > stop_frame:
                return start_seq_idx, end_seq_idx
            
            has_people = check_if_has_people(self.model, frames)
            if has_people:
                last_seen_people = self.main_frame_count
                seen_people_at_least_once = True
                print('people')
        
        return start_seq_idx, end_seq_idx
    
    def save_sequence(self, start_idx: int, end_idx: int, dst: str, save_callback: Callable):
        self.set_position(start_idx)
        for i in range(start_idx, end_idx):
            ret, frames = self.read_frames()
            if not ret:
                break
            save_callback(dst, i, frames)


    def set_position(self, pos: int):
        for i, vid in enumerate(self.videos):
            self.frame_counts[i] = pos * self.fps_list[i] // self.main_fps
            vid.set(cv2.CAP_PROP_POS_FRAMES, self.frame_counts[i])
    
        self.main_frame_count = pos

    def read_frames(self):
        ret = True
        frames = []

        for i in range(len(self.frame_counts)):
            ret_i, frame = self.videos[i].read()
            self.frame_counts[i] += 1
            while self.frame_counts[i] * self.main_fps / self.fps_list[i] <= self.main_frame_count:
                ret_i, frame = self.videos[i].read()
                self.frame_counts[i] += 1
            
            frames.append(frame)
            ret &= ret_i

        self.main_frame_count += 1
        return ret, frames            


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


def save_frames(dst: str, frame_count: int, frames: List[np.ndarray]):
    os.makedirs(os.path.join(dst, '1'), exist_ok=True)
    os.makedirs(os.path.join(dst, '2.1'), exist_ok=True)
    os.makedirs(os.path.join(dst, '2.2'), exist_ok=True)
    
    frame1, frame2 = frames
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
