from moviepy.editor import VideoFileClip
from carnd_advanced_lane_detection import ROOT_DIR
import os

SOURCE_VIDEO_PATH = os.path.join(ROOT_DIR, 'project_video.mp4')
TARGET_VIDEO_DIRECTORY = os.path.join(ROOT_DIR, 'test_videos')


def save_excerpts(clip, clip_inds):
    # beginning, left turn
    if 'b' in clip_inds:
        subclip_beginning = clip.subclip(0, 5)
        subclip_beginning.write_videofile(os.path.join(TARGET_VIDEO_DIRECTORY, 'beginning_5_sec.mp4'))

    # middle, straight
    if 'm' in clip_inds:
        subclip_middle = clip.subclip(18, 23)
        subclip_middle.write_videofile(os.path.join(TARGET_VIDEO_DIRECTORY, 'middle_5_sec.mp4'))

    # end, right, straight
    if 'e' in clip_inds:
        subclip_end = clip.subclip(37, 42)
        subclip_end.write_videofile(os.path.join(TARGET_VIDEO_DIRECTORY, 'end_5_sec.mp4'))


if __name__ == "__main__":
    clip = VideoFileClip(SOURCE_VIDEO_PATH)
    save_excerpts(clip, ['m'])