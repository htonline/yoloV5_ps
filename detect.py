# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

"""detect.py是用来预测一张图片/视频文件夹
    # --source: 传入待预测的图片文件
    执行命令: python detect.py --source data\\images\\bus.jpg
"""

"""导入库"""
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

"""定义了一些路径"""
#   __file__: 当前文件的路径;
#   FILE: 转换成绝对路径
FILE = Path(__file__).resolve()                   # D:\WorkFile\myPythonProject\deepLearning\yolov5-master\detect.py
#   ROOT: 获取当前文件的父目录
ROOT = FILE.parents[0]  # YOLOv5 root directory,    D:\WorkFile\myPythonProject\deepLearning\yolov5-master

#   确保YoLoV5的项目路径是在模块的查询路径中的！
#   sys.path: 模块的查询路径列表->后面导入模块的时候，能找到的原因就是因为这里面存在了yolov5的文件路径;
#   如果没有, 后面导包就会出错, 因为它找不到这个文件路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH; 如果不存在, 就添加进去,确保后面导包顺利。

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative 将root目录的绝对路径转换成相对路径

"""导入相对路径下的一些模块"""
from models.common import DetectMultiBackend    # 导入models文件夹下的common.py中的DetectMultiBackend类
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

"""以上导包代码执行完后, 会跳转到if __name__ == '__main__'执行 """


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    """分为六部分"""
    """1.处理预测路径: 对传入的source: "data\\images\\bus.jpg" 进行一个额外的判断"""
    source = str(source)    # python detect.py --source data\\images\\bus.jpg. 强制将路径转换成字符串类型.

    # 判断文件是否需要保存下来
    # nosave: 一个参数:False; not nosave : True;
    # source.endswith('.txt'): 判断文件是否是以txt结尾的, 否的话,说明结果是需要保存下来的;
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # 判断传入的路径是不是一个文件地址
    # source: "data\\images\\bus.jpg"
    # Path(source).suffix: ".jpg"       获取后缀
    # Path(source).suffix[1:]: "jpg"
    # in (IMG_FORMATS + VID_FORMATS):    判断jpg是否在这两个变量中: 图片格式和视频格式
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    # 判断地址是否是一个网络流/网络图片地址;
    # lower(): 全部小写
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # 判断地址是不是一个摄像头/.txt/文件/网络流地址?
    # isnumeric: 判断它是不是数值;  (--source 0: 表示打开电脑上的一个摄像头)
    # source.endswith('.txt'): 判断是不是文件
    # is_url: 是不是网络流
    # is_file: 是不是文件地址
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    screenshot = source.lower().startswith('screen')

    # 判断是不是网络流地址, 并且是不是文件; 如果是的话，就去下载图片/视频
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    """2.新建一个保存结果的文件夹"""
    # Path(project) / name : runs/detect/exp
    # increment_path: 增量路径. 统计目录下的exp文件夹到数字几了, 会自动增加文件夹exp1, exp2 ...
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run

    # 如果传入参数的时候，传入了save_txt这个参数， 就会在该文件夹下，创建labels文件夹，用来存放txt的结果。
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    """3.加载模型的权重"""
    # 选择加载模型的设备: GPU/CPU
    device = select_device(device)

    # 选择模型的后端框架. Multi:多; Backend:后端;
    # PyTorch,TorchScript, dnn, ...
    # weights: 模型权重; 参数传的是yolov5s, 其他的还有yolov5m,yolov5x, yolov5n,yolov5l ...
    # device: GPU/CPU
    # data: 一个文件的路径
    # half: 半精度推理过程（没用到）
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)

    # 获取模型的步长(一般是32), 模型能检测出来的类别名, 模型是不是pytorch
    stride, names, pt = model.stride, model.names, model.pt

    # 检测步长是不是32的倍数, 如果是的话, 模型大小就还是640x640; 如果不是,就默认计算一个倍数,修改后的图片倍数
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    """4.加载待预测的图片"""
    bs = 1                  # batch_size 每次输入一张图片
    if webcam:              # 判断地址是不是一个摄像头/.txt/文件/网络流地址?-False
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        # 加载图片文件
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    """5.模型推理过程: 将图片输入模型, 产生一个预测结果, 将检测框画出来."""

    # warmup(热身): 内部初始化了一张空白的图片, 让模型进行一次前馈传播,相当于先随便给GPU一张图片,让它跑一下->热身
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))

    # 存储一些中间的结果信息
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # 将自己的图片依次传给模型,让它依次去进行预测
    # path: 加载图片路径
    # im: resize后的图片(3,640,480) 3通道,高,宽
    # im0s: 原图(1080, 810)
    # vid_cap: null
    # s: 字符串信息, 方便输出打印
    for path, im, im0s, vid_cap, s in dataset:
        # 1.对每张图片进行预处理
        with dt[0]:
            # 将numpy形式的数组 转成pytorch支持的tensor格式, 以便输入模型中; 然后放到GPU/CPU上
            im = torch.from_numpy(im).to(model.device) # torch.Size([3,640,480])
            # 判断模型是否用到半精度
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 将图片的像素点除以255(归一化操作)
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 判断图片维度是否为3
            if len(im.shape) == 3:
                # 扩增一个维度batch
                im = im[None]  # expand for batch dim. torch.Size([1, 3, 640, 480])

        # Inference
        # 2.预测
        with dt[1]:
            # 参数visualize: 如果传过来的是true, 则会在模型推断的过程中, 把中间的特征图也保存下来。
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False

            # 参数augment: 如果传过来的是true, 推断时, 是否要做一个数据增强(可能会对推断结果有帮助,但也会降低模型运行速度)
            # pred得到的是模型预测出来的所有检测框 torch.Size([1, 18900, 85])
            # 一共18900个框. 85=4个坐标信息+1个置信度信息+80个类别的概率值
            pred = model(im, augment=augment, visualize=visualize)

        # NMS(non_max_suppression)
        # 3.非极大值抑制
        with dt[2]:
            # 非极大值过滤. 根据置信度conf_thres和iou_thres, 进行过滤
            # max_det: 一张图里, 最大能检测出来多少个目标, 如果超过,就会自动过滤掉剩下的目标;
            # 结果pred: torch.Size([1,5,6]): 5个检测框;每个检测框都有6个信息=4[检测框(左上角坐标,右下角坐标)]+1[置信度(概率)]+1[目标类别]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        # 将所有检测框画到原图中, 并保存结果
        # det: 5个检测框的预测信息
        for i, det in enumerate(pred):  # per image （遍历每个图片）
            seen += 1   # 图片计数
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # 图片保存路径 p.name 图片名
            save_path = str(save_dir / p.name)  # im.jpg
            # txt保存路径 默认不存->不管
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # 给s又拼接了一些信息, 图片尺寸
            s += '%gx%g ' % im.shape[2:]  # print string

            # 获得原图的宽和高的大小.
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # 是否要把检测框检测出来的那部分裁剪出来,单独保存.
            imc = im0.copy() if save_crop else im0  # for save_crop

            # 绘图工具(原图, 线条框, 预测的标签名)
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # det:5个框, 如果有框的话, 就画出来
            if len(det):
                # Rescale boxes from img_size to im0 size
                # im:[640,480].
                # 预测值是基于640x480的图像去预测的, 所以预测出来的值不能直接画到原图中, 需要做一个坐标映射,方便到原图中画框
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 遍历每个框, 统计框的类别.
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 是否保存预测结果
                for *xyxy, conf, cls in reversed(det):
                    # 结果保存为txt（默认不执行）
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 保存为图片
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # 隐藏标签hide_labels
                        # 隐藏置信度hide_conf: 概率那个值
                        # 如果都为false, 就会画标签名和置信度
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # 画
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    # 是否将目标框截下来, 保存为图片. 默认false
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            # 返回画好框后的图片
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # 将图片显示为一个窗口展示
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)     # opencv保存图片函数
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    """6.最终打印输出信息"""
    # seen: 统计总共有多少张图片
    # dt: 总共耗时
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image 统计每张图片的平均时间
    # 打印耗时信息
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    # 如果将结果保存为txt或图片, 额外打印图片保存地址等
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    """1.定义命令行可以传入的参数"""
    #   default: 默认值; 如果在命令行没有传入这个参数的话, 就采用默认值。
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()

    """2.对imgSize这个参数进行一个额外的判断"""
    # 判断长度是否为1, 如果是的话, 就乘以2: 640 640
    # 默认值是[640], 即长度为1;
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    """3.将所有参数信息打印出来, 并返回"""
    print_args(vars(opt))
    return opt


def main(opt):
    # requirements.txt文件中会有一些python依赖包
    # 该函数就是检测这些包有没有成功安装
    check_requirements(exclude=('tensorboard', 'thop'))

    # 后续图片的加载、预测、结果保存等一系列流程在run中执行
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()       # 解析命令行传入的参数: "--source data\\images\\bus.jpg"
    main(opt)               # 执行自定义的main函数
