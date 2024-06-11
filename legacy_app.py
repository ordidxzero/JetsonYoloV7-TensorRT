import sys
import cv2
import imutils
from yoloDet import YoloTRT
import argparse

# use path for library and engine file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="image", help="YOLO mode: image or video"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="yolov7/images/K-016235-027733-029667-031885_0_2_0_2_90_000_200.png",
        help="image to detect objects",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="videos/testvideo.mp4",
        help="video to detect objects",
    )
    parser.add_argument("--conf", type=float, default=0.9, help="confidence threshold")

    opt = parser.parse_args()

    conf = opt.conf
    yolo_mode = opt.mode

    model = YoloTRT(
        library="yolov7/build/libmyplugins.so",
        engine="yolov7/build/yolov7-tiny.engine",
        conf=conf,
        yolo_ver="v7",
    )

    if yolo_mode == "image":
        img = cv2.imread(opt.image)
        det_res, det_set, t, img = model.Inference(img)
        print(det_set)
        cv2.imshow("Output", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        video = opt.video
        if video.isdigit():
            video = int(video)
        cap = cv2.VideoCapture(video)
        while True:
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=600)
            det_res, det_set, t, frame = model.Inference(frame)
            print(det_set)
            cv2.imshow("Output", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
