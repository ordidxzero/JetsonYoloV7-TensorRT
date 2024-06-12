#!/usr/bin/env python3
#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the 'Software'),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
import sys
import threading
from multiprocessing import Process
import traceback
import numpy as np

from yoloDet import YoloTRT
import pycuda.driver as cuda


class Stream(Process):
    def __init__(self, conf=0.9, **kwargs):
        super(Stream, self).__init__()
        self.conf = conf
        self.pipe = kwargs.get("pipe")

        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()

        self.model = YoloTRT(
            library="yolov7/build/libmyplugins.so",
            engine="yolov7/build/yolov7-tiny.engine",
            conf=self.conf,
            yolo_ver="v7",
        )

    def __del__(self):
        self.ctx.pop()

    def run(self):
        while True:
            data = self.pipe.recv()

            if data == "exit":
                self.ctx.pop()
                break

            try:
                data = np.asarray(data)
                _, det_set, _, img = self.model.Inference(data)
                self.pipe.send((det_set, img))
            except:
                traceback.print_exc()
                self.pipe.send("error")
                break
