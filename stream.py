from multiprocessing import Process
import traceback
import numpy as np

from yoloDet import YoloTRT

# import pycuda.driver as cuda


class Stream(Process):
    def __init__(self, conf=0.9, **kwargs):
        super(Stream, self).__init__()
        self.conf = conf
        self.pipe = kwargs.get("pipe")
        self.daemon = True

        # cuda.init()
        # self.device = cuda.Device(0)
        # self.ctx = self.device.make_context()

        self.model = YoloTRT(
            library="yolov7/build/libmyplugins.so",
            engine="yolov7/build/yolov7-tiny.engine",
            conf=self.conf,
            yolo_ver="v7",
        )

    # def __del__(self):
    #     self.ctx.pop()

    def run(self):
        while True:
            data = self.pipe.recv()

            # if data == "exit":
            #     self.ctx.pop()
            #     break

            try:
                data = np.asarray(data)
                _, det_set, _, img = self.model.Inference(data)
                self.pipe.send((det_set, img))
            except:
                traceback.print_exc()
                self.pipe.send("error")
                break
