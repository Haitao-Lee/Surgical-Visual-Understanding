from ultralytics import YOLOv10
import random
import time
import os
import cv2


def get_certain_files_in_folder(folder_path, file_type):
    """
    Returns a list containing the full paths of all certain files within the specified folder.
    
    Parameters:
    folder_path (str): Path to the folder containing certain files.
    
    Returns:
    list: List of full paths of all certain files found.
    """
    # Use glob module to find all .mp4 files' paths within the given folder
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(file_type):
                file_paths.append(os.path.join(root, f).replace('\\','/'))
    return file_paths


def save_frames_from_video(video_path, output_folder, interal=600):
    # Check if the output folder exists, if not, create it
    start = time.time()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    frame_count = 0
    while True:
        # Read each frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        # Break the loop if no frame is returned (end of video)
        if not ret:
            break     
        # Define the filename for the frame and save it as a jpg
        frame_filename = os.path.join(output_folder, f"slice_nr_{frame_count}_.jpg")
        cv2.imwrite(frame_filename, frame)    
        # Print to confirm the frame is saved
        # print(f"Saved {frame_filename}")
        frame_count += interal
    # Release the video capture object
    cap.release()
    end = time.time()
    print("Video processing completed:", end-start)


# Load a pretrained YOLOv10n model
model = YOLOv10("./best.pt")
# save_frames_from_video('./test/case_010_video_part_002.mp4', './case_010_video_part_002')
# Perform object detection on an image
# results = model("test1.jpg")
train_img_dirs = get_certain_files_in_folder('././case_010_video_part_002', '.jpg')
random.shuffle(train_img_dirs)
for dire in train_img_dirs:
    start = time.time()
    results = model.predict(dire) #r"C:\LHT_MICCAI_Challenge2024\EndoVis\detection\data\vott-json-export\case_154_video_part_001.mp4#t=14508.566667.jpg")
    results[0].show()
    # Display the results
    end = time.time()
    print(end-start)
