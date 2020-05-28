import cv2
import numpy as np
from app.tuning import RotateProcessor
from app import preprocess_utils
from app.dto.preprocess_config import Config

import tensorflow as tf
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("heleo")
rotate_processor = RotateProcessor()

# CLASS_NAME = [0,90,180,270]
# @伟一的方向是和我这边反着的，重新调整类别,191205
CLASS_NAME = [0, 270, 180, 90]
ANGLE_MAP = {
    0: 0,
    1: 3,
    2: 2,
    3: 1
}


def init_log():
    logger.info("hello")
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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


def pred_batch(config: Config):
    model_path = "./model/100001"
    img_path_txt = "data/validate.txt"

    img_lines = open(img_path_txt).readlines()
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # init = tf.global_variables_initializer()
    # sess.run(init)
    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.allow_growth = True
    with tf.Session(config=gpuConfig) as sess:
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
            logger.info("------- %r", f_path.split(" "))
            [im_fn, img_angle] = f_path.split(" ")

            image = cv2.imread(im_fn)
            cnt_all += 1
            # 预测
            logger.info("image shape:%r",image.shape)
            # angle, img_rotate = tuning(image)
            # logger.info("小角度：%r", angle)
            img_rotate = image
            if config.do_crop_edge:
                img_rotate = crop_image_edge(img_rotate, config.crop_edge_percent)
            #TODO 图片太大或太小缩放图片到合理大小
            # 短边1600，长边 2600
            if config.do_resize:
                w,h,_ = img_rotate.shape
                min_val = min(w,h)
                # 设一个最大最小值
                if min_val < 1200 or min_val > 1800:
                    x_scale = 1600/min_val
                    logger.info("缩放倍数：%r",x_scale)
                    img_rotate = cv2.resize(img_rotate, None, fx=x_scale, fy=x_scale, interpolation=cv2.INTER_AREA)
                    logger.info("resize之后 image shape:%r",img_rotate.shape)

            patches = preprocess_utils.get_patches(img_rotate, config)
            # logger.debug("将图像分成%d个patches", len(patches))
            logger.info("开始预测")
            candiCls = sess.run(output, feed_dict={input_x: patches})
            logger.info("预测结束：%r", candiCls)
            print("预测结束：",candiCls)
            # 返回众数
            counts = np.bincount(candiCls)
            cls = np.argmax(counts)
            angle = CLASS_NAME[cls]
            # TODO 验证
            real_cls = ANGLE_MAP.get(cls)
            if str(real_cls) == img_angle:
                true_cnt += 1
                logger.info("预测正确")
            else:
                logger.info("预测错误")
            logger.info("预测角度：%r,%r,真实角度：%r,已预测条数：%r，正确条数：%r", cls, real_cls, img_angle,cnt_all,true_cnt)
            #
            # if cls == 0:
            #     rotate_image = image
            # elif cls == 1:
            #     rotate_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            # elif cls == 2:
            #     rotate_image = cv2.rotate(image, cv2.ROTATE_180)
            # elif cls == 3:
            #     rotate_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        logger.info("--------------end------------------------")
        logger.info("模式[%r""]预测结束：总条数：%r,正确条数：%r,，正确率:%r",config.name, cnt_all, true_cnt, true_cnt / cnt_all)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    config1 = Config()
    config1.name = "默认"
    pred_batch(config1)
    #
    # config2 = Config()
    # config2.name = "无标准化"
    # config2.do_std = False
    # pred_batch(config2)
    #
    # config3 = Config()
    # config3.name = "切除边缘%5,标准化"
    # config3.do_crop_edge = True
    # pred_batch(config3)
    #
    # config4 = Config()
    # config4.name = "切除边缘%5,无标准化"
    # config4.do_crop_edge = True
    # # pred_batch(config4)
    #
    # config5 = Config()
    # config5.name = "nms最小20最大2000"
    # # config5.nms_min_area = 20
    # # config5.nms_max_area = 2000
    # config5.do_debug = True
    # pred_batch(config5)
    #
    # config6 = Config()
    # config6.name = "nms最小200最大2000 & iou0.5"
    # config6.nms_min_area = 200
    # config6.nms_max_area = 2000
    # config6.nms_iou = 0.5
    # pred_batch(config6)


if __name__ == '__main__':
    logger.debug("testset")
    main()
