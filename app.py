from flask import Flask, request
from flask_restful import Api, Resource
from flask_cors import CORS
import requests
from api import get_item_info
from werkzeug.utils import secure_filename
from threading import Thread
import sys
import cv2
import imutils
from yoloDet import YoloTRT
from stream import Stream
from multiprocessing import Pipe
import os
import pycuda.driver as cuda
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)
api = Api(app)

parent_conn, child_conn = Pipe()
stream = Stream(conf=0.9, pipe=child_conn)


class FileUpload(Resource):
    def post(self):

        # model = YoloTRT(
        #     library="yolov7/build/libmyplugins.so",
        #     engine="yolov7/build/yolov7-tiny.engine",
        #     conf=0.9,
        #     yolo_ver="v7",
        # )

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

        # img_path = os.path.join("./test_images", filename)
        # file.save(img_path)

        # img = cv2.imread(img_path)
        # det_res, det_set, t, img = model.Inference(img)

        parent_conn.send(img)

        det_set, _ = parent_conn.recv()

        results = []
        img_width, img_height = img.size

        threads = [None] * len(det_set)
        results = [None] * len(det_set)

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


api.add_resource(FileUpload, "/upload")

if __name__ == "__main__":
    stream.start()
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=False)
