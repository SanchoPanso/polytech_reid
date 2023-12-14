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

    print(cam1_path)
    print(cam2_path)

    print(video1.get(cv2.CAP_PROP_FPS))
    print(video2.get(cv2.CAP_PROP_FPS))


    print(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video2.get(cv2.CAP_PROP_FRAME_COUNT))
    



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', type=str, default='/home/student2/Downloads/Pair-1/')
    parser.add_argument('--dst', type=str, default='/home/student2/Downloads/Pair-1-samples/')

    args = parser.parse_args()
    return args


def create_samples(video: cv2.VideoCapture, fps: float, dst_dir: str):
    pass

if __name__ == '__main__' :
    main()
    
    # src_dir = '/home/student2/Downloads/Pair-1/'
    # dst_dir = '/home/student2/Downloads/Pair-1-samples'


    
    # # fn = '1384-video-Cam1.mp4'
    # # name = os.path.splitext(fn)[0]
    # # cur_dst_dir = os.path.join(dst_dir, name)
    # # os.makedirs(cur_dst_dir, exist_ok=True)

    # # video_path = os.path.join(src_dir, fn)
    # # video = cv2.VideoCapture(video_path)
    
    # # cnt = 0
    # # while True:
    # #     ret, frame = video.read()
    # #     if ret is False:
    # #         break

    # #     frame_path = os.path.join(cur_dst_dir, f"{name}_{cnt // 2}.jpg")
    # #     print(frame_path)

    # #     if cnt % 2 == 0:
    # #         cv2.imwrite(frame_path, frame)
    # #     cnt += 1


    # # video.release()




    # fn = '1385-video-Cam2.mp4'
    # name = os.path.splitext(fn)[0]
    # cur_dst_dir = os.path.join(dst_dir, name)
    # os.makedirs(cur_dst_dir + '.1', exist_ok=True)
    # os.makedirs(cur_dst_dir + '.2', exist_ok=True)

    # video_path = os.path.join(src_dir, fn)
    # video = cv2.VideoCapture(video_path)
    
    # cnt = 0
    # while True:
    #     ret, frame = video.read()
    #     if ret is False:
    #         break
        
    #     for i in range(2):
            
    #         start_idx = len(frame) // 2 * i
    #         stop_idx = len(frame) // 2 * (i + 1)
    #         subframe = frame[start_idx: stop_idx]
            
    #         subframe_path = os.path.join(cur_dst_dir + f'.{i + 1}', f"{name}.{i + 1}_{cnt}.jpg")
    #         print(subframe_path)

    #         cv2.imwrite(subframe_path, subframe)

    #     cnt += 1


    # video.release()
