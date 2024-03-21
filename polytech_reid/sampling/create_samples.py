import os
import cv2
import time
import argparse
import glob


def main():
    args = parse_args()

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
    
    fps_gain = fps2 / fps1
    
    print('cam1_path:', cam1_path)
    print('cam2_path:', cam2_path)

    print('fps1:', video1.get(cv2.CAP_PROP_FPS))
    print('fps2:', video2.get(cv2.CAP_PROP_FPS))

    print('frame_count_1:', video1.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame_count_2:', video2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    create_samples_1(video1, cam1_name, fps_gain, args.dst, args.frame_limit)
    create_samples_2(video2, cam2_name, args.dst, args.frame_limit)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', type=str, default=r'D:\datasets\reid\polytech\Pair-1')
    parser.add_argument('--dst', type=str, default=r'D:\datasets\reid\polytech\Pair-1')
    parser.add_argument('--frame_limit', type=int, default=2600)

    args = parser.parse_args()
    return args


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
            if not os.path.exists(frame_path):
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
            if not os.path.exists(subframe_path):
                cv2.imwrite(subframe_path, subframe)
            
            print(cnt, subframe_path)
        
    video.release()


if __name__ == '__main__' :
    main()
