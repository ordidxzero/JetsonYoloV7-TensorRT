# 0. Install Apache2
Jeston Nano를 HTTP Server로서 사용하기 위해서 `apache2`를 설치합니다.
```shell
$ sudo apt-get update
$ sudo apt-get install apache2
$ sudo service apache2 start
$ ifconfig
```

`ifconfig`를 통해서 `eth0`의 아이피 주소를 확인해주세요.

> HYU-wlan, HY-WiFi로는 접속이 불가능합니다.

# 1. Install Libraries
```shell
$ sudo apt-get update
$ sudo apt-get install -y liblapack-dev libblas-dev gfortran libfreetype6-dev libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev
$ sudo apt-get install -y python3-pip
```

# 2. Install below python packages
flask 관련된 패키지들을 추가로 설치해야합니다.
```shell
$ numpy==1.19.0
$ pandas==0.22.0
$ Pillow==8.4.0
$ PyYAML==3.12
$ scipy==1.5.4
$ psutil
$ tqdm==4.64.1
$ imutils
$ Flask
$ Flask-RESTful
$ flask-cors
$ xmltodict
```

# 3. Install PyCuda
```shell
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
python3 -m pip install pycuda --user
```
	

# 4. Install Seaborn
```shell
$ sudo apt install python3-seaborn
```
# 5. Install torch & torchvision
```shell
$ wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
$ pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
$ git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
$ cd torchvision
$ sudo python3 setup.py install
```
# 6. Generate wts file from pt file
```shell
$ python3 gen_wts.py -w yolov7-tiny.pt -o yolov7-tiny.wts
```
# 7. Make
```shell
$ cd yolov7/
$ mkdir build
$ cd build
$ cp ../../yolov7-tiny.wts .
$ cmake ..
$ make 
```
# 8. Build Engine file 
```shell
$ sudo ./yolov7 -s yolov7-tiny.wts  yolov7-tiny.engine t
```
# 9. Python Object Detection
```shell
$ python3 app.py
```

- ip: `ifconfig`를 통해서 확인한 `eth0`의 아이피 주소

`http://{ip}:5000`으로 통신이 가능합니다.

## API Docs

1. [POST] `/upload`
   - body에 key는 `image`, value는 이미지 파일을 담아 요청하면 됩니다.