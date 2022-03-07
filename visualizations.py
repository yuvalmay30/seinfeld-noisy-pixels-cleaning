import numpy as np
import cv2 as cv


def display_images_as_video(frames: np.ndarray, wait_key=False) -> None:
  wait_key = 0 if wait_key else 50
  
  for frame in frames:
    cv.imshow('frame', frame)
    cv.waitKey(wait_key)


def display_side_by_side_video(first_video: np.ndarray, second_video: np.ndarray, is_horizontal=False,
                               wait_key=False) -> None:
  side_by_side_video = []
  for first_video_frame, second_video_frame in zip(first_video, second_video):
    side_by_side_video_frame = get_side_by_side_image(first_video_frame, second_video_frame, is_horizontal)
    side_by_side_video.append(side_by_side_video_frame)
  
  side_by_side_video = np.array(side_by_side_video)
  display_images_as_video(side_by_side_video, wait_key)


def get_side_by_side_video(first_video: np.ndarray, second_video: np.ndarray, is_horizontal=True) -> np.ndarray:
  video = []
  
  for i, (first_video_frame, second_video_frame) in enumerate(zip(first_video, second_video)):
    side_by_side_frames = get_side_by_side_image(first_video_frame, second_video_frame, is_horizontal=is_horizontal)
    video.append(side_by_side_frames)
  
  video = np.array(video)
  return video


def get_side_by_side_image(first_image: np.ndarray, second_image: np.ndarray, is_horizontal=True) -> np.ndarray:
  if is_horizontal:
    side_by_side = np.hstack((first_image, second_image))
  else:
    side_by_side = np.vstack((first_image, second_image))
  return side_by_side


def display_image(image: np.ndarray) -> None:
  cv.imshow('image', image)
  cv.waitKey(0)

