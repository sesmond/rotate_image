'''
params:
    image: the input image
    params:
     'hough_params':{
                'is_user_hough':    true,   # 是否用霍夫变换
                'line_max':         100,    # 霍夫线最大值
                'line_min':         1,      # 霍夫线最小值
                'force':            false,  # 是否强制使用行列
                'force_force':      false,  # 是否强制使用行列切必须行列达到指定数 ???
                'force_row':        false,  # 是否强制行
                'cvt_threshold1':   300,    # 灰度轮廓阀值1
                'cvt_threshold2':   600     # 灰度轮廓阀值2
        }
return:(angle,r_image)
    angle: the rotate angle
    r_image: the rotated image
'''
def process(image,params):
    return 0,image