from flask import Flask, request
from flask_restful import Api, Resource
from flask_cors import CORS
from api import get_item_info
from threading import Thread
import cv2
from yoloDet import YoloTRT
from stream import Stream
from multiprocessing import Pipe
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--multiprocessing", action="store_true")
parser.add_argument(
    "--no-multiprocessing", dest="multiprocessing", action="store_false"
)
parser.set_defaults(multiprocessing=False)

opt = parser.parse_args()

MULTIPROCESSING_MODE = opt.multiprocessing

app = Flask(__name__)
CORS(app)
api = Api(app)

if MULTIPROCESSING_MODE:
    parent_conn, child_conn = Pipe()
    stream = Stream(conf=0.9, pipe=child_conn)


class FileUpload(Resource):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = kwargs.get("model", None)

    def post(self):

        file = request.files["image"]

        filestr = file.read()
        _, ext = file.filename.split(".")

        if ext.lower() == "jpg":
            ext = "jpeg"

        # convert string data to numpy array
        npimg = np.frombuffer(filestr, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = Image.fromarray(img.astype("uint8"))

        img_width, img_height = img.size

        det_set = dict()

        # ==================== Multiprocessing Mode ===================
        if MULTIPROCESSING_MODE:
            parent_conn.send(img)
            det_set, _ = parent_conn.recv()

        # ================ End of  Multiprocessing Mode ===============

        # ==================== Single Process Mode ====================
        if not MULTIPROCESSING_MODE:
            if self.model == None:
                self.model = YoloTRT(
                    library="yolov7/build/libmyplugins.so",
                    engine="yolov7/build/yolov7-tiny.engine",
                    conf=0.9,
                    yolo_ver="v7",
                )

            img = np.array(img)

            _, det_set, _, _ = self.model.Inference(img)

        # ================ End of Single Process Mode =================

        threads = [None] * len(det_set)
        results = [None] * len(det_set)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for i, cls in enumerate(det_set):
            threads[i] = Thread(
                target=get_item_info, args=(det_set[cls], img, ext, results, i)
            )
            threads[i].start()

        for i in range(len(threads)):
            threads[i].join()

        results = [x for x in results if x is not None]

        return {
            "data": sorted(results, key=lambda x: x["item_name"]),
            "img_height": img_height,
            "img_width": img_width,
        }


if not MULTIPROCESSING_MODE:
    model = YoloTRT(
        library="yolov7/build/libmyplugins.so",
        engine="yolov7/build/yolov7-tiny.engine",
        conf=0.9,
        yolo_ver="v7",
    )
    api.add_resource(FileUpload, "/upload", resource_class_kwargs={"model": model})


if MULTIPROCESSING_MODE:
    api.add_resource(FileUpload, "/upload")
    stream.start()

app.run(debug=False, host="0.0.0.0", port=5000, threaded=False)
