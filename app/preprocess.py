from functools import reduce
import cv2
import numpy as np
import logging
from app.tuning import RotateProcessor
from app import preprocess_utils
from app.dto.preprocess_config import Config

import tensorflow as tf

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

rotate_processor = RotateProcessor()

logger = logging.getLogger(__name__)

# CLASS_NAME = [0,90,180,270]
# @伟一的方向是和我这边反着的，重新调整类别,191205
CLASS_NAME = [0, 270, 180, 90]
ANGLE_MAP = {
    0: 0,
    1: 3,
    2: 2,
    3: 1
}


# img 是字符串数组
def blur_detect(img):
    # 如果不做blur校验，就直接返回(not blur)
    if not conf.CFG.preprocess_blur: return False

    """
    :param path: image path
    :return: (flag, score)
            flag: 1 means the image is sharp, 0 otherwise
    """
    T = 0.15
    block_size = 16

    # 做一个上三角阵，16x16
    weight = reduce(lambda x, y: x + y,
                    [(np.eye(block_size, block_size, i) +
                      np.eye(block_size, block_size, -i)) *
                     (block_size - i)
                     for i in range(1, block_size)])
    weight += np.eye(block_size, block_size) * block_size

    weight_sum = np.sum(weight)
    # img = cv2.imread(path, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = np.zeros((block_size, block_size), dtype=int)
    h, w = img.shape
    vNum = int(h / block_size)
    hNum = int(w / block_size)

    # 按照纵向移动16x16模板
    for vIdx in range(vNum):
        for hIdx in range(hNum):
            vStart = vIdx * block_size
            hStart = hIdx * block_size

            # 切出图片来
            patch = img[vStart:(vStart + block_size), hStart:(hStart + block_size)]
            patch = np.float32(patch)

            # 对图做离散余弦变换
            patchDCT = cv2.dct(patch)
            patchDCT = np.abs(patchDCT)

            # 看最高高频量，是否大于所有量的1/5?
            # 这个是一个经验值么？
            # 是说这个块非常的明锐么？
            if patchDCT.max() > patchDCT.sum() / 5:
                continue

            # 看大于8的值的个数，hist是（16x16）
            # 也就是把那些高频的去掉后的，低频对应的傅里叶结果图中值大于8的，累加到一起
            patchDCTValid = patchDCT > 8
            # patchDCTValid = patchDCT > (patchDCT.max()*0.001)
            hist += patchDCTValid.astype(int)

    blurValid = hist > (0.3 * hist[0, 0])
    s = (np.multiply(blurValid.astype(int), weight)).sum() / weight_sum

    if s > T:
        return False  # not blur，清晰
    else:
        return True  # blur，模糊


# 判断是否是单据图像
def valid_detect(img, is_debug=False):
    # 如果不判断是否是短句，就直接返回是True：合法
    if not conf.CFG.preprocess_valid: return True

    T = 0.6  # 白像素占比阈值
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # 根据图像的尺寸判断，768*1366是典型的屏幕分辨率，基本可以确定是屏幕截图，1920*1080有待观察，不能完全确定
    if (h == 768 and w == 1366) or \
            (h == 720 and w == 1280) or \
            (w == 768 and h == 1366) or \
            (w == 720 and h == 1280):
        logger.debug("图像不是单据，原因：长宽不合规：h=%d,w=%d", h, w)
        return False
    # 太小的图像
    if h * w < 4e5:
        if is_debug: print("h*w<5e5", h * w)
        logger.debug("图像不是单据，原因：h*w[%f]<5e5", h * w)
        return False
    whiteRatio = np.mean(gray == 255)
    # 白像素占比过高，比较确定是无效图片
    if whiteRatio < T: return True

    logger.debug("图像不是单据，原因：白像素占比过高:%f", whiteRatio)
    return False


# 返回的是以原图逆时针旋转"多少度？"后变正
def tuning(image):
    # 如果不做微调，就直接返回是0度和原图
    angle, img_rotate = rotate_processor.process(image)
    logger.debug("微调旋转角度为：%f", angle)
    return angle, img_rotate


def crop_image_edge(image, percent):
    """
    切除图片的边缘
    :param image:
    :param percent: 边缘百分比
    :return:
    """
    w, h, _ = image.shape
    w_c = int(w * percent)
    h_c = int(h * percent)

    new_img = image[w_c:w - w_c, h_c:h - h_c]
    return new_img


def main(config: Config):
    model_path = "./model/100001"
    img_path_txt = "data/validate.txt"

    img_lines = open(img_path_txt).readlines()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    with sess:
        # 从pb模型直接恢复

        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
        signature = meta_graph_def.signature_def
        in_tensor_name = signature['serving_default'].inputs['x'].name
        out_tensor_name = signature['serving_default'].outputs['predCls'].name

        input_x = sess.graph.get_tensor_by_name(in_tensor_name)
        output = sess.graph.get_tensor_by_name(out_tensor_name)
        # 遍历所有图片预测
        # config = Config()
        cnt_all = 0
        true_cnt = 0
        for im_line in img_lines:
            f_path = im_line.rstrip('\n')
            # print("f_path:", f_path)
            print("-------", f_path.split(" "))
            [im_fn, img_angle] = f_path.split(" ")

            image = cv2.imread(im_fn)
            cnt_all += 1
            # 预测
            print("image shape:",image.shape)
            angle, img_rotate = tuning(image)
            if config.do_crop_edge:
                img_rotate = crop_image_edge(img_rotate, config.crop_edge_percent)

            print("小角度：", angle)
            patches = preprocess_utils.get_patches(img_rotate, config)
            # logger.debug("将图像分成%d个patches", len(patches))
            print("开始预测")
            candiCls = sess.run(output, feed_dict={input_x: patches})
            print("预测结束：", candiCls)
            # 返回众数
            counts = np.bincount(candiCls)
            cls = np.argmax(counts)
            angle = CLASS_NAME[cls]
            print("预测角度：", cls, angle, ",真实角度：", img_angle)
            # TODO 验证
            real_cls = ANGLE_MAP.get(cls)
            if str(real_cls) == img_angle:
                true_cnt += 1
                print("预测正确")
            else:
                print("预测错误")
            #
            # if cls == 0:
            #     rotate_image = image
            # elif cls == 1:
            #     rotate_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            # elif cls == 2:
            #     rotate_image = cv2.rotate(image, cv2.ROTATE_180)
            # elif cls == 3:
            #     rotate_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print("--------------end------------------------")
        print("模式 预测结束：总条数：", cnt_all, ",正确条数：", true_cnt, "，正确率：", true_cnt / cnt_all)


if __name__ == '__main__':
    config1 = Config()
    main(config1)
    # config2 = Config()
    # main(config2)
