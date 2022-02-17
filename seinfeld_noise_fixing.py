from typing import Tuple

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def load_video() -> Tuple[np.ndarray, cv.VideoCapture]:
    frames = []
    video_capture = cv.VideoCapture('Videos/seinfeld_s03_e01.mp4')
    
    while video_capture.isOpened():
        is_succeed, frame = video_capture.read()
    
        if not is_succeed:
            print("Can't receive frame - probably loaded all frames")
            break
    
        frames.append(frame)
    
    frames = np.array(frames)
    return frames, video_capture


def display_images_as_video(frames: np.ndarray) -> None:
    for frame in frames:
        cv.imshow('frame', frame)
        cv.waitKey(50)
      

def get_side_by_side_image(first_image: np.ndarray, second_image: np.ndarray, is_horizontal=True) -> np.ndarray:
    if is_horizontal:
        side_by_side = np.hstack((first_image, second_image))
    else:
        side_by_side = np.vstack((first_image, second_image))
    return side_by_side


def display_image(image: np.ndarray) -> None:
    cv.imshow('image', image)
    cv.waitKey(0)


def display_median_filter_result(frames: np.ndarray) -> None:
    # display_images_as_video(frames)
    
    median_frames = np.zeros_like(frames)
    for idx, frame in enumerate(frames):
        median_frame = cv.medianBlur(frame, 5)
        median_frames[idx] = median_frame
        
        image = get_side_by_side_image(frame, median_frame, is_horizontal=False)
        display_image(image)
      
    # display_images_as_video(median_frames)


def release_resources(video_capture: cv.VideoCapture):
    video_capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    frames, video_capture = load_video()
    first_frames = frames[:5]

    _, H, W, _ = first_frames.shape

    half_frames = first_frames[:, :int(H * 3 / 7), :int(W * 2 / 3), :]
    # display_images_as_video(half_frames)

    display_median_filter_result(half_frames)
    # median_frame = cv.medianBlur(first_frames[0], 11)
    # display_images_side_by_side(first_frames[0], median_frame)

    release_resources(video_capture)
