import requests
import xmltodict
from PIL import Image
from io import BytesIO
import base64


def get_url(item_seq):
    serviceKey = "EKtNW6TT0KaIR80WfezIL%2F2Jll2KDTPvDBD1LI9WwZ4BQzo0ibySTLvwtDk3fisRd57f8IhdIwNMiT5GmWxp5w%3D%3D"
    url = f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnDtlInq04?serviceKey={serviceKey}&type=json"
    return f"{url}&item_seq={item_seq}"


def parse_ee_doc(EE_DOC_DATA):
    parsed = xmltodict.parse(EE_DOC_DATA, encoding="utf-8")["DOC"]
    article = parsed["SECTION"]["ARTICLE"]

    if isinstance(article, list):
        effects = list(map(lambda x: x["@title"], article))
    else:
        effects = list(map(lambda x: x["#text"], article["PARAGRAPH"]))
    return effects


def mapping_effect(x):
    title = x["@title"]
    paragraphs = x["PARAGRAPH"]

    effects = []
    if isinstance(paragraphs, dict):
        effects.append(paragraphs["#text"])
    else:
        effects = list(map(lambda y: y["#text"], paragraphs))

    return {
        "title": title,
        "effects": effects,
    }


def parse_nb_doc(NB_DOC_DATA):
    parsed = xmltodict.parse(NB_DOC_DATA, encoding="utf-8")["DOC"]
    article = parsed["SECTION"]["ARTICLE"]
    effects = list(
        map(
            mapping_effect,
            article,
        )
    )
    return effects


def parse_response(res):

    if res.status_code != 200:
        raise Exception("Failed to fetch data")

    res = res.json()

    body = res["body"]

    if body["totalCount"] == 0:
        return None

    item = body["items"][0]

    item_name = item["ITEM_NAME"]  # 이름
    entp_name = item["ENTP_NAME"]  # 제조사
    etc_otc_code = item["ETC_OTC_CODE"]  # 일반의약품
    storage_method = item["STORAGE_METHOD"].strip()  # 기밀용기, 실온(1～30℃)보관
    valid_term = item["VALID_TERM"]  # 제조일로부터 36 개월
    EE_DOC_DATA = parse_ee_doc(item["EE_DOC_DATA"])  # 효능효과
    NB_DOC_DATA = parse_nb_doc(item["NB_DOC_DATA"])  # 사용상의주의사항

    return {
        "item_name": item_name,
        "entp_name": entp_name,
        "etc_otc_code": etc_otc_code,
        "storage_method": storage_method,
        "valid_term": valid_term,
        "EE_DOC_DATA": EE_DOC_DATA,
        "NB_DOC_DATA": NB_DOC_DATA,
    }


def get_base64_from(img: Image, format="JPEG"):
    rawBytes = BytesIO()
    img.save(rawBytes, format)
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.getvalue())
    img_base64 = (
        bytes(f"data:image/{format.lower()};base64,", encoding="utf-8") + img_base64
    )
    img_str = img_base64.decode("utf-8")
    return img_str


def get_item_info(det, img, ext, result, idx):
    item_seq = det["item_seq"]
    box = det["box"]

    x1, y2, x2, y1 = box

    img_crop = img.crop((x1, y1, x2, y2))

    img_str = get_base64_from(img_crop, format=ext)

    url = get_url(item_seq)
    res = requests.get(url)
    parsed = parse_response(res)

    if parsed is not None:
        parsed["img_base64"] = img_str
        result[idx] = parsed


# url = get_url(201802815)
# url = get_url(200607849)
# url = get_url(200500251)
# url = get_url(200300416)


# res = requests.get(url)
# parsed = parse_response(res)

# if parsed is not None:
#     print(json.dumps(parsed, indent=4, ensure_ascii=False))
# else:
#     print("No data")
