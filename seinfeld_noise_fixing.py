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


def display_following_frame_filter_results(frames: np.ndarray, black_threshold=50, white_threshold=200) -> None:
  minimal_blackish_or_whitish_channels = 3
  H, W, _ = frames[0].shape
  # display_images_as_video(frames)
  
  fixed_frames = np.zeros_like(frames)
  for idx, frame in enumerate(frames):
    next_frame_index = idx + 1
    if next_frame_index is len(frames):
      break

    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    dark_pixels_mask = frame_hsv[:, :, 2] < black_threshold

    bright_pixels_mask = frame_hsv[:, :, 2] > white_threshold

    dark_and_bright_mask = dark_pixels_mask | bright_pixels_mask
    mask = dark_and_bright_mask

    # blackish_pixels_mask = frame < black_threshold
    # number_of_blackish_channels_in_pixel = np.count_nonzero(blackish_pixels_mask, axis=2)
    # minimal_blackish_pixels_mask = number_of_blackish_channels_in_pixel >= minimal_blackish_or_whitish_channels
    #
    # whitish_pixels_mask = frame > white_threshold
    # number_of_whitish_channels_in_pixel = np.count_nonzero(whitish_pixels_mask, axis=2)
    # minimal_whitish_pixels_mask = number_of_whitish_channels_in_pixel >= minimal_blackish_or_whitish_channels
    # blackish_and_whitish_mask = minimal_blackish_pixels_mask | minimal_whitish_pixels_mask
   
    # mask = blackish_and_whitish_mask

    masked_frame = np.copy(frame)
    masked_frame[mask] = 255, 255, 255
    masked_frame[~mask] = 0, 0, 0
   
    image = get_side_by_side_image(frame, masked_frame, is_horizontal=False)
    display_image(image)
    
    # next_frame = frames[next_frame_index]
    #
    # fixed_frame = np.copy(frame)
    # fixed_frame[mask] = next_frame[mask]
    #
    # fixed_frames[idx] = fixed_frame
    #
    # image = get_side_by_side_image(frame, fixed_frame, is_horizontal=False)
    # display_image(image)
    #
  # display_images_as_video(fixed_frames)


def release_resources(video_capture: cv.VideoCapture):
  video_capture.release()
  cv.destroyAllWindows()


if __name__ == '__main__':
  frames, video_capture = load_video()
  first_frames = frames[:5]
  
  _, H, W, _ = first_frames.shape
  
  original_frames = first_frames[:, :int(H * 3 / 7), :int(W * 2 / 3), :]

  fixed_frames = original_frames
  # fixed_frames = display_gaussian_filter_result(fixed_frames)
  fixed_frames = display_bilateral_filter_result(fixed_frames)
  fixed_frames = display_bilateral_filter_result(fixed_frames)
  # fixed_frames = display_median_filter_result(fixed_frames, 3)
  # display_mean_filter_result(fixed_frames)
  # display_following_frame_filter_results(fixed_frames)
  
  display_side_by_side_video(original_frames, fixed_frames, wait_key=True)
  # display_side_by_side_video(original_frames, fixed_frames)
  
  cv.imwrite('Results/image.png', get_side_by_side_image(original_frames[0], fixed_frames[0],
                                                                             is_horizontal=False))
  
  release_resources(video_capture)
