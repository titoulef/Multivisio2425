from trackers import PlayerTracker
import multivisio
import cv2

def main():
    input_video_path1 = ('input_videos/hall3.mp4')
    input_video_path2 = ('input_videos/hall2.mp4')
    #show
    multivisio.loop(input_video_path1, fpsDivider=5, videoScale=1)
    #multivisio.loop2(input_video_path1, input_video_path2, fpsDivider=10, videoScale=1)

if __name__ == '__main__':
    main()