from flask import Flask, request
from flask_restful import Api, Resource
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

app = Flask(__name__)
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

        filename = secure_filename(file.filename)

        img_path = os.path.join("./test_images", filename)
        file.save(img_path)

        img = cv2.imread(img_path)
        # det_res, det_set, t, img = model.Inference(img)

        parent_conn.send(img)

        det_set, img = parent_conn.recv()

        return {"data": det_set, "img_height": img.shape[0], "img_width": img.shape[1]}


api.add_resource(FileUpload, "/upload")

if __name__ == "__main__":
    stream.start()
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=False)
