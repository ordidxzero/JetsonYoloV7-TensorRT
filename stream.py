from multiprocessing import Process
import traceback
import numpy as np

from yoloDet import YoloTRT


class Stream(Process):
    def __init__(self, conf=0.9, **kwargs):
        super(Stream, self).__init__()
        self.conf = conf

        # * flask 프로세스와 통신할 Pipe
        self.pipe = kwargs.get("pipe")

        # * daemon 프로세스로 설정
        self.daemon = True

        self.model = YoloTRT(
            library="yolov7/build/libmyplugins.so",
            engine="yolov7/build/yolov7-tiny.engine",
            conf=self.conf,
            yolo_ver="v7",
        )

    def run(self):
        while True:
            # * flask에서 이미지를 넘겨줄 경우 실행
            data = self.pipe.recv()

            try:
                data = np.asarray(data)
                _, det_set, _, img = self.model.Inference(data)

                # * 추론 결과를 Pipe를 통해 반환
                self.pipe.send((det_set, img))
            except:
                traceback.print_exc()
                self.pipe.send("error")
                break
