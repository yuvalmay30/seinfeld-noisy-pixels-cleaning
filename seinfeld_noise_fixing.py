from typing import Tuple

import numpy as np
import cv2 as cv


def load_video() -> Tuple[np.ndarray, cv.VideoCapture]:
  frames = []
  video_capture = cv.VideoCapture('Videos/seinfeld_s03_e01_2.mp4')
  
  while video_capture.isOpened():
    is_succeed, frame = video_capture.read()
    
    if not is_succeed:
      print("Can't receive frame - probably loaded all frames")
      break
    
    frames.append(frame)
  
  frames = np.array(frames)
  return frames, video_capture


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
  

def get_side_by_side_image(first_image: np.ndarray, second_image: np.ndarray, is_horizontal=True) -> np.ndarray:
  if is_horizontal:
    side_by_side = np.hstack((first_image, second_image))
  else:
    side_by_side = np.vstack((first_image, second_image))
  return side_by_side


def display_image(image: np.ndarray) -> None:
  cv.imshow('image', image)
  cv.waitKey(0)


def display_median_filter_result(frames: np.ndarray, filter_size=3) -> np.ndarray:
  # display_images_as_video(frames)
  
  fixed_frames = np.zeros_like(frames)
  for idx, frame in enumerate(frames):
    median_frame = cv.medianBlur(frame, filter_size)
    fixed_frames[idx] = median_frame
    
    # image = get_side_by_side_image(frame, median_frame, is_horizontal=False)
    # display_image(image)

  # display_side_by_side_video(frames, fixed_frames, is_horizontal=False)
  return fixed_frames


def display_mean_filter_result(frames: np.ndarray, filter_size=3) -> np.ndarray:
  # display_images_as_video(frames)
  
  fixed_frames = np.zeros_like(frames)
  for idx, frame in enumerate(frames):
    fixed_frame = cv.blur(frame, (filter_size, filter_size), 0)
    fixed_frames[idx] = fixed_frame
    
    # image = get_side_by_side_image(frame, fixed_frame, is_horizontal=False)
    # display_image(image)
  
  # display_side_by_side_video(frames, fixed_frames, is_horizontal=False)
  return fixed_frames


def display_gaussian_filter_result(frames: np.ndarray, filter_size=5) -> np.ndarray:
  # display_images_as_video(frames)
  
  fixed_frames = np.zeros_like(frames)
  for idx, frame in enumerate(frames):
    fixed_frame = cv.GaussianBlur(frame, (filter_size, filter_size), 0)
    fixed_frames[idx] = fixed_frame
    
    # image = get_side_by_side_image(frame, fixed_frame, is_horizontal=False)
    # display_image(image)
  
  # display_side_by_side_video(frames, fixed_frames, is_horizontal=False)
  return fixed_frames


def display_bilateral_filter_result(frames: np.ndarray, filter_size=-1) -> np.ndarray:
  # display_images_as_video(frames)
  
  fixed_frames = np.zeros_like(frames)
  for idx, frame in enumerate(frames):
    fixed_frame = cv.bilateralFilter(frame, filter_size, 5, 13)
    fixed_frames[idx] = fixed_frame
    
    # image = get_side_by_side_image(frame, fixed_frame, is_horizontal=False)
    # display_image(image)
  
  # display_side_by_side_video(frames, fixed_frames, is_horizontal=False)
  return fixed_frames


def display_prev_next_median_filter_result(frames: np.ndarray) -> np.ndarray:
  # display_images_as_video(frames)
  
  number_of_prev_and_next_filters = 2
  fixed_frames = np.zeros_like(frames)

  for idx in range(number_of_prev_and_next_filters, len(frames)-number_of_prev_and_next_filters-1):
    current_layer = frames[idx - number_of_prev_and_next_filters: idx + number_of_prev_and_next_filters + 1]
    median = np.median(current_layer, axis=0)
    fixed_frame = median
    fixed_frames[idx] = fixed_frame
    
    # image = get_side_by_side_image(frame, fixed_frame, is_horizontal=False)
    # display_image(image)
  
  # display_side_by_side_video(frames, fixed_frames, is_horizontal=False)
  return fixed_frames


def release_resources(video_capture: cv.VideoCapture):
  video_capture.release()
  cv.destroyAllWindows()


if __name__ == '__main__':
  frames, video_capture = load_video()
  first_frames = frames[:200]
  
  _, H, W, _ = first_frames.shape
  
  original_frames = first_frames[:, :int(H * 3 / 7), :int(W * 2 / 3), :]

  fixed_frames = original_frames
  # fixed_frames = display_gaussian_filter_result(fixed_frames)
  # fixed_frames = display_bilateral_filter_result(fixed_frames)
  # fixed_frames = display_median_filter_result(fixed_frames, 3)
  fixed_frames = display_prev_next_median_filter_result(fixed_frames)
  # display_mean_filter_result(fixed_frames)
  # display_following_frame_filter_results(fixed_frames)
  
  # display_side_by_side_video(original_frames, fixed_frames, wait_key=True)
  display_side_by_side_video(original_frames, fixed_frames)
  
  output_image = get_side_by_side_image(original_frames[2], fixed_frames[2], is_horizontal=False)
  cv.imwrite('Results/image.png', output_image)
  
  release_resources(video_capture)
