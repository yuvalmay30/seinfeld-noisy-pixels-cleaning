from typing import Tuple

import cv2
from scipy.ndimage import gaussian_filter1d, gaussian_filter as sp_gaussian_filter
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


def median_filter(frames: np.ndarray, filter_size=3) -> np.ndarray:
  fixed_frames = np.zeros_like(frames)
  for idx, frame in enumerate(frames):
    median_frame = cv.medianBlur(frame, filter_size)
    fixed_frames[idx] = median_frame

  return fixed_frames


def mean_filter(frames: np.ndarray, filter_size=3) -> np.ndarray:
  fixed_frames = np.zeros_like(frames)
  for idx, frame in enumerate(frames):
    fixed_frame = cv.blur(frame, (filter_size, filter_size), 0)
    fixed_frames[idx] = fixed_frame
    
  return fixed_frames


def gaussian_filter(frames: np.ndarray, filter_size=5) -> np.ndarray:
  fixed_frames = np.zeros_like(frames)
  for idx, frame in enumerate(frames):
    fixed_frame = cv.GaussianBlur(frame, (filter_size, filter_size), 0)
    fixed_frames[idx] = fixed_frame
  
  return fixed_frames


def bilateral_filter(frames: np.ndarray, filter_size=-1) -> np.ndarray:
  fixed_frames = np.zeros_like(frames)
  for idx, frame in enumerate(frames):
    fixed_frame = cv.bilateralFilter(frame, filter_size, 5, 13)
    fixed_frames[idx] = fixed_frame
    
  return fixed_frames


def prev_next_average_filter(frames: np.ndarray) -> np.ndarray:
  number_of_prev_and_next_filters = 1
  fixed_frames = np.zeros_like(frames)

  for idx in range(number_of_prev_and_next_filters, len(frames)-number_of_prev_and_next_filters):
    current_layer = frames[idx - number_of_prev_and_next_filters: idx + number_of_prev_and_next_filters + 1]
    fixed_frame = np.average(current_layer, axis=0)
    fixed_frames[idx] = fixed_frame

  return fixed_frames


def prev_next_median_filter(frames: np.ndarray) -> np.ndarray:
  number_of_prev_and_next_filters = 1
  fixed_frames = np.zeros_like(frames)

  for idx in range(number_of_prev_and_next_filters, len(frames)-number_of_prev_and_next_filters):
    current_layer = frames[idx - number_of_prev_and_next_filters: idx + number_of_prev_and_next_filters + 1]
    fixed_frame = np.median(current_layer, axis=0)
    fixed_frames[idx] = fixed_frame

  return fixed_frames


def prev_next_gaussian_filter(frames: np.ndarray) -> np.ndarray:
  number_of_prev_and_next_filters = 1
  fixed_frames = np.zeros_like(frames)

  for idx in range(number_of_prev_and_next_filters, len(frames)-number_of_prev_and_next_filters):
    current_layer: np.ndarray = frames[idx - number_of_prev_and_next_filters: idx + number_of_prev_and_next_filters + 1]
    current_layer = np.moveaxis(current_layer, 0, 3)
    
    fixed_frame = gaussian_filter1d(current_layer, sigma=5, axis=3)
    fixed_frame = fixed_frame[:, :, :, 1]
    fixed_frames[idx] = fixed_frame

  return fixed_frames


def prev_next_median_filter_with_moving_objects_filter(frames: np.ndarray) -> np.ndarray:
  moving_object_difference_threshold = 40
  
  number_of_median_prev_and_next_filters = 2
  fixed_frames = np.zeros_like(frames)
  
  for idx in range(number_of_median_prev_and_next_filters, len(frames) - number_of_median_prev_and_next_filters):
    current_median_layer = frames[idx - number_of_median_prev_and_next_filters:
                                  idx + number_of_median_prev_and_next_filters + 1]
    
    median_frame = np.median(current_median_layer, axis=0)
    fixed_frame = median_frame
    
    if idx > number_of_median_prev_and_next_filters:
      frame = frames[idx]
      
      frames_diff = frame - fixed_frame
      abs_diff = np.abs(frames_diff)
      sum_abs_diff = np.sum(abs_diff, axis=2)
      
      big_diff_mask = sum_abs_diff > moving_object_difference_threshold
      
      # filtered_frame = cv.medianBlur(frame, 3)
      # filtered_frame = cv.bilateralFilter(frame, 3, 5, 5)
      # averaged_frame[big_diff_mask] = filtered_frame[big_diff_mask]
      
      fixed_frame[big_diff_mask] = frame[big_diff_mask]
      
      # big_diff_mask_as_float = big_diff_mask.astype(float)
      # big_diff_mask_as_float = big_diff_mask_as_float * 255.
      # cv.imshow('diff', big_diff_mask_as_float)
      # cv.waitKey(0)
      # cv.imwrite('./Results/average_mask.png', big_diff_mask_as_float)
    
    fixed_frames[idx] = fixed_frame
  
  return fixed_frames


def prev_next_filter_with_moving_objects_filter(frames: np.ndarray) -> np.ndarray:
  moving_object_difference_threshold = 10
  
  number_of_median_prev_and_next_filters = 2
  number_of_average_prev_and_next_filters = 1
  averaged_frames = np.zeros_like(frames)
  fixed_frames = np.zeros_like(frames)

  for idx in range(number_of_median_prev_and_next_filters, len(frames)-number_of_median_prev_and_next_filters):
    current_median_layer = frames[idx - number_of_median_prev_and_next_filters:
                                  idx + number_of_median_prev_and_next_filters + 1]
    current_average_layer = frames[idx - number_of_average_prev_and_next_filters:
                                   idx + number_of_average_prev_and_next_filters + 1]

    averaged_frame = np.average(current_average_layer, axis=0)
    median_frame = np.median(current_median_layer, axis=0)

    fixed_frame = median_frame

    if idx > number_of_average_prev_and_next_filters:
      frame = frames[idx]

      frames_diff = averaged_frame - averaged_frames[idx-1]
      abs_diff = np.abs(frames_diff)
      sum_abs_diff = np.sum(abs_diff, axis=2)

      big_diff_mask = sum_abs_diff > moving_object_difference_threshold

      # filtered_frame = cv.medianBlur(frame, 3)
      # filtered_frame = cv.bilateralFilter(frame, 3, 5, 5)
      # averaged_frame[big_diff_mask] = filtered_frame[big_diff_mask]
      
      averaged_frame[big_diff_mask] = frame[big_diff_mask]
      fixed_frame[big_diff_mask] = frame[big_diff_mask]
      
      # big_diff_mask_as_float = big_diff_mask.astype(float)
      # big_diff_mask_as_float = big_diff_mask_as_float * 255.
      # cv.imshow('diff', big_diff_mask_as_float)
      # cv.waitKey(0)
      # cv.imwrite('./Results/average_mask.png', big_diff_mask_as_float)

    averaged_frames[idx] = averaged_frame
    fixed_frames[idx] = fixed_frame
  
  fixed_frames[:number_of_median_prev_and_next_filters] = frames[:number_of_median_prev_and_next_filters]
  fixed_frames[-number_of_median_prev_and_next_filters:] = frames[-number_of_median_prev_and_next_filters:]

  return fixed_frames


def release_resources(video_capture: cv.VideoCapture):
  video_capture.release()
  cv.destroyAllWindows()


def write_video(video: np.ndarray):
  output_file_path = 'Results/video3.mp4'
  codec = cv2.VideoWriter_fourcc(*'mp4v')
  FPS = 20.0
  _, frame_height, frame_width, _ = video.shape
  
  video_writer = cv2.VideoWriter(output_file_path, codec, FPS, (frame_width, frame_height))
  
  for frame in video:
    video_writer.write(frame)
  
  video_writer.release()
  

if __name__ == '__main__':
  frames, video_capture = load_video()
  first_frames = frames
    
  _, H, W, _ = first_frames.shape
  
  # original_frames = first_frames[:, :int(H * 3 / 7), :int(W * 2 / 3), :]
  original_frames = first_frames

  fixed_frames = original_frames
  # fixed_frames = gaussian_filter(fixed_frames)
  # fixed_frames = bilateral_filter(fixed_frames)
  # fixed_frames = bilateral_filter(fixed_frames)
  # fixed_frames = median_filter(fixed_frames, 5)
  # fixed_frames = mean_filter(fixed_frames, 5)

  # fixed_frames = prev_next_average_filter(fixed_frames)
  # fixed_frames = prev_next_median_filter(fixed_frames)
  # fixed_frames = prev_next_gaussian_filter(fixed_frames)
  # display_side_by_side_video(first, second, wait_key=True)

  # fixed_frames = prev_next_median_filter_with_moving_objects_filter(fixed_frames)
  fixed_frames = prev_next_filter_with_moving_objects_filter(fixed_frames)
  # display_mean_filter_result(fixed_frames)
  # display_following_frame_filter_results(fixed_frames)
  
  # display_side_by_side_video(original_frames, fixed_frames, wait_key=True)
  # display_side_by_side_video(original_frames, fixed_frames)
  
  output_image = get_side_by_side_image(original_frames[3], fixed_frames[3], is_horizontal=False)
  cv.imwrite('Results/image.png', output_image)

  side_by_side_video = get_side_by_side_video(original_frames, fixed_frames, is_horizontal=False)
  # write_video(side_by_side_video)
  write_video(fixed_frames)

  release_resources(video_capture)
