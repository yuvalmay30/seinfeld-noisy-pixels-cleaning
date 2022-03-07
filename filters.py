from scipy.ndimage import gaussian_filter1d
import numpy as np
import cv2 as cv


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


def average_filter_on_consecutive_frames(frames: np.ndarray) -> np.ndarray:
  number_of_prev_and_next_filters = 1
  fixed_frames = np.zeros_like(frames)
  
  for idx in range(number_of_prev_and_next_filters, len(frames) - number_of_prev_and_next_filters):
    current_layer = frames[idx - number_of_prev_and_next_filters: idx + number_of_prev_and_next_filters + 1]
    fixed_frame = np.average(current_layer, axis=0)
    fixed_frames[idx] = fixed_frame
  
  return fixed_frames


def median_filter_on_consecutive_frames(frames: np.ndarray) -> np.ndarray:
  number_of_prev_and_next_filters = 1
  fixed_frames = np.zeros_like(frames)
  
  for idx in range(number_of_prev_and_next_filters, len(frames) - number_of_prev_and_next_filters):
    current_layer = frames[idx - number_of_prev_and_next_filters: idx + number_of_prev_and_next_filters + 1]
    fixed_frame = np.median(current_layer, axis=0)
    fixed_frames[idx] = fixed_frame
  
  return fixed_frames


def gaussian_filter_on_consecutive_frames(frames: np.ndarray) -> np.ndarray:
  number_of_prev_and_next_filters = 1
  fixed_frames = np.zeros_like(frames)
  
  for idx in range(number_of_prev_and_next_filters, len(frames) - number_of_prev_and_next_filters):
    current_layer: np.ndarray = frames[idx - number_of_prev_and_next_filters: idx + number_of_prev_and_next_filters + 1]
    current_layer = np.moveaxis(current_layer, 0, 3)
    
    fixed_frame = gaussian_filter1d(current_layer, sigma=5, axis=3)
    fixed_frame = fixed_frame[:, :, :, 1]
    fixed_frames[idx] = fixed_frame
  
  return fixed_frames


def median_filter_on_consecutive_frames_with_median_mask(frames: np.ndarray) -> np.ndarray:
  moving_object_difference_threshold = 10
  
  number_of_median_prev_and_next_filters = 2
  fixed_frames = np.zeros_like(frames)
  
  for idx in range(number_of_median_prev_and_next_filters, len(frames) - number_of_median_prev_and_next_filters):
    current_median_layer = frames[idx - number_of_median_prev_and_next_filters:
                                  idx + number_of_median_prev_and_next_filters + 1]
    
    fixed_frame = np.median(current_median_layer, axis=0)
    
    if idx > number_of_median_prev_and_next_filters:
      frame = frames[idx]
      
      frames_diff = frame - fixed_frame
      abs_diff = np.abs(frames_diff)
      sum_abs_diff = np.sum(abs_diff, axis=2)
      
      big_diff_mask = sum_abs_diff > moving_object_difference_threshold
      fixed_frame[big_diff_mask] = frame[big_diff_mask]
    
    fixed_frames[idx] = fixed_frame
  
  return fixed_frames


def median_filter_on_consecutive_frames_with_average_mask(frames: np.ndarray) -> np.ndarray:
  moving_object_difference_threshold = 10
  
  number_of_median_prev_and_next_filters = 2
  number_of_average_prev_and_next_filters = 1
  averaged_frames = np.zeros_like(frames)
  fixed_frames = np.zeros_like(frames)
  
  for idx in range(number_of_median_prev_and_next_filters, len(frames) - number_of_median_prev_and_next_filters):
    current_median_layer = frames[idx - number_of_median_prev_and_next_filters:
                                  idx + number_of_median_prev_and_next_filters + 1]
    current_average_layer = frames[idx - number_of_average_prev_and_next_filters:
                                   idx + number_of_average_prev_and_next_filters + 1]
    
    averaged_frame = np.average(current_average_layer, axis=0)
    median_frame = np.median(current_median_layer, axis=0)
    
    fixed_frame = median_frame
    
    if idx > number_of_average_prev_and_next_filters:
      frame = frames[idx]
      
      frames_diff = averaged_frame - averaged_frames[idx - 1]
      abs_diff = np.abs(frames_diff)
      sum_abs_diff = np.sum(abs_diff, axis=2)
      
      big_diff_mask = sum_abs_diff > moving_object_difference_threshold
      averaged_frame[big_diff_mask] = frame[big_diff_mask]
      fixed_frame[big_diff_mask] = frame[big_diff_mask]
    
    averaged_frames[idx] = averaged_frame
    fixed_frames[idx] = fixed_frame
  
  fixed_frames[:number_of_median_prev_and_next_filters] = frames[:number_of_median_prev_and_next_filters]
  fixed_frames[-number_of_median_prev_and_next_filters:] = frames[-number_of_median_prev_and_next_filters:]
  
  return fixed_frames
