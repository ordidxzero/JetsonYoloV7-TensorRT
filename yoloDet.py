import cv2
import numpy as np
import tensorrt as trt
import random
import ctypes
import pycuda.autoinit
import pycuda.driver as cuda
import time
from PIL import ImageFont, ImageDraw, Image


font = ImageFont.truetype("fonts/MaruBuri-Bold.ttf", 14)


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
BOUNDARY_LIMIT = 50
host_inputs = []
cuda_inputs = []
host_outputs = []
cuda_outputs = []
bindings = []


class YoloTRT:
    def __init__(self, library, engine, conf, yolo_ver):
        cuda.init()
        device = cuda.Device(0)
        self.ctx = device.make_context()
        self.CONF_THRESH = conf
        self.IOU_THRESHOLD = 0.9
        self.LEN_ALL_RESULT = 38001
        self.LEN_ONE_RESULT = 38
        self.yolo_version = yolo_ver
        self.categories = [
            "K-000250",
            "K-000573",
            "K-006192",
            "K-002483",
            "K-012778",
            "K-019552",
            "K-022627",
            "K-023223",
            "K-005002",
            "K-013161",
            "K-037777",
            "K-038954",
            "K-044834",
            "K-004378",
            "K-005886",
            "K-012081",
            "K-013395",
            "K-022362",
            "K-001866",
            "K-003743",
            "K-005094",
            "K-025438",
            "K-016548",
            "K-001900",
            "K-004543",
            "K-003544",
            "K-016551",
            "K-006563",
            "K-021026",
            "K-027926",
            "K-029345",
            "K-029451",
            "K-031705",
            "K-033009",
            "K-010224",
            "K-018110",
            "K-019607",
            "K-021771",
            "K-024850",
            "K-033208",
            "K-044199",
            "K-003351",
            "K-016232",
            "K-003832",
            "K-016262",
            "K-016688",
            "K-020238",
            "K-021325",
            "K-022074",
            "K-029667",
            "K-035206",
            "K-036637",
            "K-038162",
            "K-013900",
            "K-018147",
            "K-018357",
            "K-019232",
            "K-020014",
            "K-031863",
            "K-032310",
            "K-033880",
            "K-041768",
            "K-022347",
            "K-003483",
            "K-019861",
            "K-028763",
            "K-030308",
            "K-031885",
            "K-034597",
            "K-020877",
            "K-025367",
            "K-025469",
            "K-027653",
            "K-027733",
            "K-027777",
            "K-010221",
            "K-012247",
            "K-012420",
            "K-023203",
            "K-027993",
            "K-029871",
            "K-033878",
            "K-003614",
            "K-005000",
            "K-005391",
            "K-015280",
            "K-016206",
            "K-020259",
            "K-020852",
            "K-024752",
            "K-028424",
            "K-033026",
            "K-023319",
            "K-006697",
            "K-006835",
            "K-020004",
            "K-025200",
            "K-026993",
            "K-029711",
            "K-053384",
            "K-030850",
            "K-038927",
            "K-041149",
            "K-043233",
            "K-044266",
            "K-010487",
            "K-012638",
            "K-013004",
            "K-020753",
            "K-019881",
            "K-024941",
            "K-026455",
            "K-035659",
            "K-038972",
            "K-011220",
            "K-015710",
            "K-016235",
            "K-038723",
        ]

        self.mapping_hangeul = {
            "K-000250": {"name": "마그밀정(수산화마그네슘)", "item_seq": 197400246},
            "K-000573": {"name": "게보린정 300mg/PTP", "item_seq": 197900277},
            "K-006192": {"name": "삐콤씨에프정 618.6mg/병", "item_seq": 199700756},
            "K-002483": {"name": "뮤테란캡슐 100mg", "item_seq": 198801531},
            "K-012778": {"name": "다보타민큐정 10mg/병", "item_seq": 200300416},
            "K-019552": {"name": "트루비타정 60mg/병", "item_seq": 200607734},
            "K-022627": {"name": "메가파워정 90mg/병", "item_seq": 200712481},
            "K-023223": {"name": "비타비백정 100mg/병", "item_seq": 200802213},
            "K-005002": {"name": "엘도스캡슐(에르도스테인)", "item_seq": 199500435},
            "K-013161": {"name": "엘스테인캡슐(에르도스테인)", "item_seq": 200301252},
            "K-037777": {"name": "알바스테인캡슐(에르도스테인)", "item_seq": 201507696},
            "K-038954": {"name": "뮤코원캡슐(에르도스테인)", "item_seq": 201602679},
            "K-044834": {"name": "엘스테인정(에르도스테인)", "item_seq": 201804394},
            "K-004378": {"cls": 13, "name": "타이레놀정500mg", "item_seq": 199303108},
            "K-005886": {
                "name": "타이레놀이알서방정(아세트아미노펜)",
                "item_seq": 199603003,
            },
            "K-012081": {"name": "리렉스펜정 300mg/PTP", "item_seq": 200202194},
            "K-013395": {"name": "써스펜8시간이알서방정 650mg", "item_seq": 200301818},
            "K-022362": {"name": "맥시부펜이알정 300mg", "item_seq": 200710817},
            "K-001866": {"name": "알마겔정(알마게이트)", "item_seq": 198700430},
            "K-003743": {"name": "알드린정", "item_seq": 199201106},
            "K-005094": {"name": "삼남건조수산화알루미늄겔정", "item_seq": 199501099},
            "K-025438": {"name": "큐시드정 31.5mg/PTP", "item_seq": 200902214},
            "K-016548": {"name": "가바토파정 100mg", "item_seq": 200500251},
            "K-001900": {"name": "보령부스파정 5mg", "item_seq": 198700706},
            "K-004543": {"name": "에어탈정(아세클로페낙)", "item_seq": 199400486},
            "K-003544": {"name": "무코스타정(레바미피드)", "item_seq": 199102825},
            "K-016551": {"name": "동아가바펜틴정 800mg", "item_seq": 200500254},
            "K-006563": {"name": "조인스정 200mg", "item_seq": 199702182},
            "K-021026": {"name": "펠루비정(펠루비프로펜)", "item_seq": 200704590},
            "K-027926": {"name": "울트라셋이알서방정", "item_seq": 201006967},
            "K-029345": {"name": "비모보정 500/20mg", "item_seq": 201200417},
            "K-029451": {"name": "레일라정", "item_seq": 201201587},
            "K-031705": {"name": "낙소졸정 500/20mg", "item_seq": 201308928},
            "K-033009": {"name": "신바로정", "item_seq": 201401963},
            "K-010224": {"name": "넥시움정 40mg", "item_seq": 200009026},
            "K-018110": {"name": "란스톤엘에프디티정 30mg", "item_seq": 200510974},
            "K-019607": {"name": "스토가정 10mg", "item_seq": 200607849},
            "K-021771": {"name": "라비에트정 20mg", "item_seq": 200708376},
            "K-024850": {"name": "놀텍정 10mg", "item_seq": 200810688},
            "K-033208": {"name": "에스원엠프정 20mg", "item_seq": 201402463},
            "K-044199": {"name": "케이캡정 50mg", "item_seq": 201802815},
            "K-003351": {"name": "일양하이트린정 2mg", "item_seq": 199101125},
            "K-016232": {"name": "리피토정 20mg", "item_seq": 200410085},
            "K-003832": {"name": "뉴로메드정(옥시라세탐)", "item_seq": 199201807},
            "K-016262": {"name": "크레스토정 20mg", "item_seq": 200410326},
            "K-016688": {
                "name": "오마코연질캡슐(오메가-3-산에틸에스테르90)",
                "item_seq": 200500497,
            },
            "K-020238": {"name": "플라빅스정 75mg", "item_seq": 200700907},
            "K-021325": {"name": "아토르바정 10mg", "item_seq": 200705992},
            "K-022074": {"name": "리피로우정 20mg", "item_seq": 200709390},
            "K-029667": {"name": "리바로정 4mg", "item_seq": 201205253},
            "K-035206": {"name": "아토젯정 10/40mg", "item_seq": 201500365},
            "K-036637": {"name": "로수젯정10/5밀리그램", "item_seq": 201503658},
            "K-038162": {"name": "로수바미브정 10/20mg", "item_seq": 201600318},
            "K-013900": {"name": "에빅사정(메만틴염산염)", "item_seq": 200308324},
            "K-018147": {"name": "리리카캡슐 150mg", "item_seq": 200511058},
            "K-018357": {
                "name": "종근당글리아티린연질캡슐(콜린알포세레이트)",
                "item_seq": 200600026,
            },
            "K-019232": {"name": "콜리네이트연질캡슐 400mg", "item_seq": 200605445},
            "K-020014": {"name": "마도파정", "item_seq": 200611524},
            "K-031863": {"name": "아질렉트정(라사길린메실산염)", "item_seq": 201309347},
            "K-032310": {"name": "글리아타민연질캡슐", "item_seq": 201310719},
            "K-033880": {"name": "글리틴정(콜린알포세레이트)", "item_seq": 201403839},
            "K-041768": {"name": "카발린캡슐 25mg", "item_seq": 201702674},
            "K-022347": {"name": "자누비아정 50mg", "item_seq": 200710758},
            "K-003483": {"name": "기넥신에프정(은행엽엑스)", "item_seq": 199102014},
            "K-019861": {"name": "노바스크정 5mg", "item_seq": 200610660},
            "K-028763": {"name": "트라젠타정(리나글립틴)", "item_seq": 201106367},
            "K-030308": {"name": "트라젠타듀오정 2.5/850mg", "item_seq": 201208291},
            "K-031885": {
                "name": "자누메트엑스알서방정 100/1000mg",
                "item_seq": 201309418,
            },
            "K-034597": {"name": "제미메트서방정 50/1000mg", "item_seq": 201405223},
            "K-020877": {"name": "엑스포지정 5/160mg", "item_seq": 200703803},
            "K-025367": {"name": "자누메트정 50/850mg", "item_seq": 200901817},
            "K-025469": {"name": "아모잘탄정 5/100mg", "item_seq": 200902349},
            "K-027653": {"name": "세비카정 10/40mg", "item_seq": 201004570},
            "K-027733": {"name": "트윈스타정 40/5mg", "item_seq": 201005081},
            "K-027777": {"name": "카나브정 60mg", "item_seq": 201005501},
            "K-010221": {"name": "쎄로켈정 100mg", "item_seq": 200009023},
            "K-012247": {"name": "아빌리파이정 10mg", "item_seq": 200202889},
            "K-012420": {"name": "자이프렉사정 2.5mg", "item_seq": 200209622},
            "K-023203": {"name": "쿠에타핀정 25mg", "item_seq": 200802023},
            "K-027993": {"name": "졸로푸트정 100mg", "item_seq": 201007672},
            "K-029871": {"name": "렉사프로정 15mg", "item_seq": 201206335},
            "K-033878": {"name": "브린텔릭스정 20mg", "item_seq": 201403837},
            "K-003614": {
                "name": "동아오팔몬정(리마프로스트알파-시클로덱스트린포접화합물)",
                "item_seq": 199200261,
            },
            "K-005000": {"name": "비유피-4정 20mg", "item_seq": 199500420},
            "K-005391": {"name": "프로스카정", "item_seq": 199504272},
            "K-015280": {"name": "한미탐스캡슐 0.2mg", "item_seq": 200402674},
            "K-016206": {"name": "아보다트연질캡슐 0.5mg", "item_seq": 200409982},
            "K-020259": {"name": "자트랄엑스엘정 10mg", "item_seq": 200701034},
            "K-020852": {"name": "베시케어정 10mg", "item_seq": 200703673},
            "K-024752": {"name": "토비애즈서방정 4mg", "item_seq": 200810113},
            "K-028424": {"name": "플리바스정 50mg", "item_seq": 201102769},
            "K-033026": {"name": "트루패스정 4mg", "item_seq": 201401986},
            "K-023319": {"name": "디카맥스디플러스정", "item_seq": 200802864},
            "K-006697": {"name": "씨즈날정(세티리진염산염)", "item_seq": 199703153},
            "K-006835": {"name": "이튼큐정 35mg/PTP", "item_seq": 199704687},
            "K-020004": {"name": "라시도필캡슐(비매품)", "item_seq": 200611395},
            "K-025200": {"name": "마이칼디정", "item_seq": 200900833},
            "K-026993": {"name": "원더칼-디츄어블정", "item_seq": 201000406},
            "K-029711": {"name": "디카테오정", "item_seq": 201205447},
            "K-053384": {"name": "디카테오엘씨정 10mg/병", "item_seq": 202009337},
            "K-030850": {"name": "비오플250캡슐 282.5mg", "item_seq": 201306134},
            "K-038927": {
                "name": "락토엔큐캡슐(바실루스리케니포르미스균)",
                "item_seq": 201602610,
            },
            "K-041149": {"name": "람노스캡슐500mg/병", "item_seq": 201700802},
            "K-043233": {"name": "바이오탑디캡슐 50mg/병", "item_seq": 201800568},
            "K-044266": {"name": "바이오탑하이캡슐 150mg/병", "item_seq": 201803000},
            "K-010487": {"name": "동화덴스톨캡슐 150mg/PTP", "item_seq": 200100780},
            "K-012638": {"name": "투스딘골드캡슐 150mg/PTP", "item_seq": 200300092},
            "K-013004": {"name": "이가탄에프캡슐 150mg/PTP", "item_seq": 200300913},
            "K-020753": {"name": "인사돌플러스정 35mg/PTP", "item_seq": 200703215},
            "K-019881": {"name": "지르텍정(세티리진염산염)", "item_seq": 200610765},
            "K-024941": {"name": "액티프롤정 60mg/PTP", "item_seq": 200811061},
            "K-026455": {"name": "클라리틴정(로라타딘)", "item_seq": 200905692},
            "K-035659": {"name": "제라타딘정(로라타딘)", "item_seq": 201501483},
            "K-038972": {"name": "알러비정(세티리진염산염)", "item_seq": 201602714},
            "K-011220": {"name": "아스피린프로텍트정 100mg", "item_seq": 200108429},
            "K-015710": {"name": "엔테론정 150mg", "item_seq": 200403500},
            "K-016235": {"name": "카듀엣정 5mg/20mg", "item_seq": 200410088},
            "K-038723": {"name": "엔트레스토필름코팅정 200mg", "item_seq": 201602113},
        }

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        ctypes.CDLL(library)

        with open(engine, "rb") as f:
            serialized_engine = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.batch_size = self.engine.max_batch_size

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.input_w = self.engine.get_binding_shape(binding)[-1]
                self.input_h = self.engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

    def __del__(self):
        self.ctx.pop()

    def PreProcessImg(self, img):
        image_raw = img
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def MappingPillCode(self, class_id):
        return self.categories[int(class_id)]

    def MappingHangeul(self, class_id):
        cls = self.MappingPillCode(class_id)
        return self.mapping_hangeul[cls]["name"]

    def MappingItemSeq(self, class_id):
        cls = self.MappingPillCode(class_id)
        return self.mapping_hangeul[cls]["item_seq"]

    def Inference(self, img):
        input_image, image_raw, origin_h, origin_w = self.PreProcessImg(img)
        np.copyto(host_inputs[0], input_image.ravel())
        stream = cuda.Stream()
        self.context = self.engine.create_execution_context()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        t1 = time.time()
        self.context.execute_async(
            self.batch_size, bindings, stream_handle=stream.handle
        )
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        t2 = time.time()
        output = host_outputs[0]

        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.PostProcess(
                output[i * self.LEN_ALL_RESULT : (i + 1) * self.LEN_ALL_RESULT],
                origin_h,
                origin_w,
            )

        det_res = []
        det_set = dict()
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            det = dict()
            cls = self.MappingPillCode(result_classid[j])
            score = result_scores[j]
            det["class"] = cls
            det["conf"] = score
            det["box"] = box
            det_res.append(det)

            if cls not in det_set:
                if (
                    box[2] - box[0] >= BOUNDARY_LIMIT
                    and box[3] - box[1] >= BOUNDARY_LIMIT
                ):
                    det_set[cls] = {
                        "box": list(map(float, box.tolist())),
                        "conf": float(score),
                        "cls_id": int(result_classid[j]),
                        "dl_name": self.MappingHangeul(result_classid[j]),
                        "item_seq": self.MappingItemSeq(result_classid[j]),
                    }
            else:
                if det_set[cls]["conf"] < float(score):
                    if (
                        box[2] - box[0] >= BOUNDARY_LIMIT
                        and box[3] - box[1] >= BOUNDARY_LIMIT
                    ):
                        det_set[cls] = {
                            "box": list(map(float, box.tolist())),
                            "conf": float(score),
                            "cls_id": int(result_classid[j]),
                            "dl_name": self.MappingHangeul(result_classid[j]),
                            "item_seq": self.MappingItemSeq(result_classid[j]),
                        }

        for j in det_set:
            box = det_set[j]["box"]
            conf = det_set[j]["conf"]
            dl_name = det_set[j]["dl_name"]
            img = self.PlotBbox(
                box,
                img,
                label="{}:  {:.2f}".format(dl_name, conf),
            )
        return det_res, det_set, t2 - t1, img

    def PostProcess(self, output, origin_h, origin_w):
        num = int(output[0])
        if self.yolo_version == "v5":
            pred = np.reshape(output[1:], (-1, self.LEN_ONE_RESULT))[:num, :]
            pred = pred[:, :6]
        elif self.yolo_version == "v7":
            pred = np.reshape(output[1:], (-1, 6))[:num, :]

        boxes = self.NonMaxSuppression(
            pred,
            origin_h,
            origin_w,
            conf_thres=self.CONF_THRESH,
            nms_thres=self.IOU_THRESHOLD,
        )
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def NonMaxSuppression(
        self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4
    ):
        boxes = prediction[prediction[:, 4] >= conf_thres]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        confs = boxes[:, 4]
        boxes = boxes[np.argsort(-confs)]
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = (
                self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            )
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

    def xywh2xyxy(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        return y

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(
            inter_rect_y2 - inter_rect_y1 + 1, 0, None
        )
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def PlotBbox(self, x, img, color=None, label=None, line_thickness=None):
        tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        )  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled

            # ==================== Add Korean Text on Image ===================== #
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            draw.text((c1[0] + 8, c1[1] - 20), label, (0, 0, 0), font=font)
            img = np.array(img_pil)
            # =================================================================== #
            # cv2.putText(
            #     img,
            #     label,
            #     (c1[0], c1[1] - 2),
            #     0,
            #     tl / 3,
            #     [225, 255, 255],
            #     thickness=tf,
            #     lineType=cv2.LINE_AA,
            # )
        return img
