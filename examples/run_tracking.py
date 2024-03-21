import os
import sys
from pathlib import Path
import glob
import yaml
from types import SimpleNamespace
from ultralytics import YOLO

from polytech_reid.trackers.tracking_wrapper import TrackingWrapper
from polytech_reid.trackers.bot_sort import BOTSORT

with open('botsort.yaml') as f:
    cfg = yaml.safe_load(f)

cfg = SimpleNamespace(**cfg)

detector = YOLO('yolov8n.pt')
tracker = BOTSORT(args=cfg, frame_rate=30)
wrapper = TrackingWrapper(detector, tracker)

img_paths = glob.glob(os.path.join('person_sequencies', '2', '1', '*'))
img_paths.sort()
wrapper.run(img_paths[:100])

