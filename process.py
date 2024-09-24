import os
import numpy as np
import cv2
from pandas import DataFrame
from pathlib import Path
from scipy.ndimage import center_of_mass, label
from pathlib import Path
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    DataFrameValidator,
)
from typing import (Tuple)
from evalutils.exceptions import ValidationError
from PIL import Image, ImageFilter
import json
from ultralytics import YOLOv10
import time


execute_in_docker = True # False # 


def calculate_intersection_area(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1[:4]
    x_min2, y_min2, x_max2, y_max2 = box2[:4]
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)
    if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
        return 0
    intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    return intersection_area

def calculate_area(box):
    x_min, y_min, x_max, y_max = box[:4]
    return (x_max - x_min) * (y_max - y_min)


def save_frames_from_video(video_path, output_folder):
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
        ret, frame = cap.read()
        # Break the loop if no frame is returned (end of video)
        if not ret:
            break     
        # Define the filename for the frame and save it as a jpg
        frame_filename = os.path.join(output_folder, f"slice_nr_{frame_count}_.jpg")
        cv2.imwrite(frame_filename, frame)    
        # Print to confirm the frame is saved
        # print(f"Saved {frame_filename}")
        frame_count += 1
    # Release the video capture object
    cap.release()
    end = time.time()
    print("Video processing completed:", end-start)


def filter_boxes(boxes):
    filtered_boxes = boxes.copy()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            box1 = boxes[i]
            box2 = boxes[j]
            if box1[5] != box2[5]:
                continue
            intersection_area = calculate_intersection_area(box1, box2)
            area1 = calculate_area(box1)
            area2 = calculate_area(box2)
            smaller_area = min(area1, area2)
            if intersection_area > 0.9 * smaller_area:
                if box1[4] > box2[4]:
                    if box2 in filtered_boxes:
                        filtered_boxes.remove(box2)
                else:
                    if box1 in filtered_boxes:
                        filtered_boxes.remove(box1)
    # while len(filtered_boxes) > 4:
    #     filtered_boxes.sort(key=lambda x: x[4])
    #     filtered_boxes.pop(0)
    return filtered_boxes


class VideoLoader():
    def load(self, *, fname):
        path = Path(fname)
        print('File found: ' + str(path))
        if ((str(path)[-3:])) == 'mp4':
            if not path.is_file():
                raise IOError(
                    f"Could not load {fname} using {self.__class__.__qualname__}."
                )
                #cap = cv2.VideoCapture(str(fname))
            #return [{"video": cap, "path": fname}]
            return [{"path": fname}]

# only path valid
    def hash_video(self, input_video):
        pass


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


class UniqueVideoValidator(DataFrameValidator):
    """
    Validates that each video in the set is unique
    """

    def validate(self, *, df: DataFrame):
        try:
            hashes = df["video"]
        except KeyError:
            raise ValidationError("Column `video` not found in DataFrame.")

        if len(set(hashes)) != len(hashes):
            raise ValidationError(
                "The videos are not unique, please submit a unique video for "
                "each case."
            )
            

class Surgtoolloc_det(DetectionAlgorithm):
    def __init__(self):
        super().__init__(
            index_key='input_video',
            file_loaders={'input_video': VideoLoader()},
            input_path=Path("/input/") if execute_in_docker else Path("./test/"),
            output_file=Path("/output/surgical-tools.json") if execute_in_docker else Path(
                            "./output/surgical-tools.json"),
            validators=dict(
                input_video=(
                    #UniqueVideoValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )       
        # Load the pretrained model
        self.model = YOLOv10("./best.pt")                                                                                                    ###
        self.tool_list = ["needle_driver",
                          "monopolar_curved_scissor",
                          "force_bipolar",
                          "clip_applier",
                          "cadiere_forceps",
                          "bipolar_forceps",
                          "vessel_sealer",
                          "permanent_cautery_hook_spatula",
                          "prograsp_forceps",
                          "stapler",
                          "grasping_retractor",
                          "tip_up_fenestrated_grasper"]

    def process_case(self, *, idx, case):
        # Input video would return the collection of all frames (cap object)
        input_video_file_path = case #VideoLoader.load(case)
        # Detect and score candidates
        # save_frames_from_video(str(case.path), './'+ os.path.splitext(os.path.basename(case.path))[0])
        scored_candidates = self.predict(case.path) #video file > load evalutils.py
        # Write resulting candidates to result.json for this case
        return dict(type="Multiple 2D bounding boxes", boxes=scored_candidates, version={"major": 1, "minor": 0})

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def predict(self, fname) -> DataFrame:
        """
        Inputs:
        fname -> video file path
        
        Output:
        tools -> list of prediction dictionaries (per frame) in the correct format as described in documentation 
        """  
        print('Video file to be loaded: ' + str(fname))
        cap = cv2.VideoCapture(str(fname))
        if not cap.isOpened():
            print("Error opening video file:" + str(fname))
        else:
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))       
            all_frames_predicted_outputs = []
            for fid in range(num_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)               
                    frame = Image.fromarray(frame)
                    width, height = frame.size
                    start_row = max(height - 50, 0)
                    top_part = frame.crop((0, 0, width, start_row))
                    bottom_part = frame.crop((0, start_row, width, height))
                    bottom_part_blurred = bottom_part.filter(ImageFilter.GaussianBlur(radius=5))
                    frame = Image.new('RGB', (width, height))
                    frame.paste(top_part, (0, 0))
                    frame.paste(bottom_part_blurred, (0, start_row))
                    init_dect = self.model(frame)
                    # init_dect[0].show()
                    init_res = filter_boxes(init_dect[0].boxes.data.to('cpu').numpy().tolist())    
                    # print(init_res)
                    predictions = []
                    for n in range(len(init_res)):
                        name = f'slice_nr_{fid}_' + self.tool_list[int(init_res[n][5])]
                        bbox = [[round(float((init_res[n][0])), 1), round(float((init_res[n][1])), 1), 0.5],
                                [round(float((init_res[n][2])), 1), round(float((init_res[n][1])), 1), 0.5],
                                [round(float((init_res[n][2])), 1), round(float((init_res[n][3])), 1), 0.5],
                                [round(float((init_res[n][0])), 1), round(float((init_res[n][3])), 1), 0.5]]
                        prediction = {"corners": bbox, "name": name, "probability": float(init_res[n][4])}
                        predictions.append(prediction)
                    # if len(predictions) > 0:
                    all_frames_predicted_outputs += predictions
            return all_frames_predicted_outputs


if __name__ == "__main__":
    Surgtoolloc_det().process()