class Config:
    #
    name = "default"
    # 小图片最多个数
    max_counter = 100
    # 是否标准化
    do_std = True
    # 多少一下的过滤掉
    nms_box_cnt = 5
    nms_min_area = 20
    nms_max_area = 600
    # 对IOU>0.3的进行NMS筛选，剩下的是靠谱的框（在remains里面）
    nms_iou = 0.3
    # 是否缩放
    do_resize = True
    # 是否去除边缘
    do_crop_edge = False
    crop_edge_percent = 0.05

    # 是否切小图存储
    do_debug = False



if __name__ == '__main__':
    print("")
