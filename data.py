from typing import Tuple
import numpy as np
import cv2 as cv


def load_video() -> Tuple[np.ndarray, cv.VideoCapture]:
  frames = []
  video_capture = cv.VideoCapture('Videos/seinfeld_s03_e01_2.mp4')
  
  while video_capture.isOpened():
    is_succeed, frame = video_capture.read()
    
    if not is_succeed:
      print("Can't load frame - probably loaded all frames")
      break
    
    frames.append(frame)
  
  frames = np.array(frames)
  return frames, video_capture


def write_video(video: np.ndarray, output_file_path):
  codec = cv.VideoWriter_fourcc(*'mp4v')
  FPS = 20.0
  _, frame_height, frame_width, _ = video.shape
  
  video_writer = cv.VideoWriter(output_file_path, codec, FPS, (frame_width, frame_height))
  
  for frame in video:
    video_writer.write(frame)
  
  video_writer.release()
