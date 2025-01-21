from trackers import PlayerTracker
import multivisio
import cv2

def main():
    input_video_path = ('input_videos/hall1.mp4')

    #show
    multivisio.loop(input_video_path, fpsDivider=4, videoScale=1)


if __name__ == '__main__':
    main()