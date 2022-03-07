import cv2 as cv

from data import load_video, write_video
from filters import median_filter_on_consecutive_frames_with_average_mask
from visualizations import display_side_by_side_video, get_side_by_side_video


def release_resources(video_capture: cv.VideoCapture):
  video_capture.release()
  cv.destroyAllWindows()
  

if __name__ == '__main__':
  frames, video_capture = load_video()
  
  fixed_frames = median_filter_on_consecutive_frames_with_average_mask(frames)

  display_side_by_side_video(frames, fixed_frames, wait_key=True)

  side_by_side_video = get_side_by_side_video(frames, fixed_frames, is_horizontal=False)
  write_video(fixed_frames, 'Results/video.mp4')

  release_resources(video_capture)
