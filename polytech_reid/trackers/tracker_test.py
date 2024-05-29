import cv2
import os
import glob
from ultralytics import YOLO
from ultralytics.engine.results import Results


def update_mot_file(frame_idx: int, results: Results, filename):
    rows = []
    result = results[0]

    if result.boxes.id is None:
        return
    
    for i, idx in enumerate(result.boxes.id):
        x, y, w, h = result.boxes.xywh[i]
        conf = result.boxes.conf[i]
        row = f"{frame_idx},{int(idx)},{(x - w/2):.2f},{(y - h/2):.2f},{w:.2f},{h:.2f},1,1,1\n"
        rows.append(row)
    
    with open(filename, 'a') as f:
        for row in rows:
            f.write(row)


# Load the YOLOv8 model
model = YOLO('yolov8s.pt')
img_paths = glob.glob(os.path.join('person_sequencies', '2', '1', '*'))
start_index = 2775

with open('gt_2_1.txt', 'w') as f:
    f.write('')

# Loop through the video frames
for i, path in enumerate(img_paths):
    # Read a frame from the video
    frame = cv2.imread(path)

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, conf=0.5, persist=True, classes=[0])

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    update_mot_file(i + 1, results, 'gt_2_1.txt')

    # Display the annotated frame
    cv2.imshow("YOLOv8 Tracking", cv2.resize(annotated_frame, (800, 600)))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
# Release the video capture object and close the display window
cv2.destroyAllWindows()
