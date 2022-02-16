# (16-726): Project 1 starter Python code
# credit to https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj1/data/colorize_skel.py
# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import cv2
import numpy as np
import skimage as sk



# align blue channel and red channel with green channel
def calculate_index(window_len, g, channel, crop, method="edge"):
    max_ncc = 0
    min_ssd = 10000
    best_i = 0
    best_j = 0

    # on height
    for i in range(-window_len, window_len):
        # on width
        for j in range(-window_len, window_len):
            channel_temp = np.roll(channel, i, axis=0)
            channel_temp = np.roll(channel_temp, j, axis=1)

            crop = int(min(channel_temp.shape) * 0.25)

            width_temp, height_temp = channel_temp.shape

            channel_temp = channel_temp[crop:height_temp - crop, crop:width_temp - crop]
            g_temp = g[crop:height_temp - crop, crop:width_temp - crop]

            # Sobel edge detection
            if method == "edge":
                channel_grad_x = cv2.Sobel(channel_temp, cv2.CV_64F, 1, 0, ksize=3)
                channel_grad_y = cv2.Sobel(channel_temp, cv2.CV_64F, 0, 1, ksize=3)
                g_grad_x = cv2.Sobel(g_temp, cv2.CV_64F, 1, 0, ksize=3)
                g_grad_y = cv2.Sobel(g_temp, cv2.CV_64F, 0, 1, ksize=3)

                channel_sobel = cv2.magnitude(channel_grad_x, channel_grad_y)
                g_sobel = cv2.magnitude(g_grad_x, g_grad_y)

                ncc = np.sum(channel_sobel * g_sobel / np.linalg.norm(channel_sobel) / np.linalg.norm(g_sobel))

                if ncc > max_ncc:
                    max_ncc = ncc
                    best_i = i
                    best_j = j


            # general ncc
            elif method == "ncc":
                ncc = np.sum(channel_temp * g_temp / np.linalg.norm(channel_temp) / np.linalg.norm(g_temp))

                if ncc > max_ncc:
                    max_ncc = ncc
                    best_i = i
                    best_j = j

            # general ssd
            elif method == "ssd":
                area = channel_temp - g_temp
                temp_height, temp_width = area.shape
                area = area[crop:temp_height - crop, crop:temp_width - crop]
                l2 = np.sum(np.square(area))

                if l2 < min_ssd:
                    min_ssd = l2
                    best_i = i
                    best_j = j

    # print("window length:", window_len)
    # print(max_i, max_j)
    # print(max_ncc)

    return best_i, best_j


def find_threshold(l, target_num=4, step=0.1, thres=0.7):
    while (len([each for each in l if each > thres]) < target_num):
        thres -= step
    return thres


def find_final_crop(img):
    left = 0
    right = 0
    top = 0
    bottom = 0

    x_sum = [each for each in np.sum(img, axis=0) / img.shape[0]]  # sum of each row
    y_sum = [each for each in np.sum(img, axis=1) / img.shape[1]]  # sum of each column

    num_thres_x = find_threshold(x_sum)
    num_thres_y = find_threshold(y_sum)

    candidate_x = [x_sum.index(each) for each in x_sum if each > num_thres_x]
    # print(candidate_x)
    candidate_y = [y_sum.index(each) for each in y_sum if each > num_thres_y]

    left = min(candidate_x)
    right = max(candidate_x)

    top = min(candidate_y)
    bottom = max(candidate_y)

    x_thres = img.shape[1] / 2
    y_thres = img.shape[0] / 2

    img_height, img_width = img.shape

    x_upper_limit = img.shape[1] / 10
    y_upper_limit = img.shape[0] / 10

    # print(x_thres, y_thres)

    for candidate in candidate_x:
        if candidate > left and candidate < x_thres and candidate < x_upper_limit:
            left = candidate
        elif candidate < right and candidate > x_thres and candidate > img_width - x_upper_limit:
            right = candidate

    for candidate in candidate_y:
        if candidate > top and candidate < y_thres and candidate < y_upper_limit:
            top = candidate
        elif candidate < bottom and candidate > y_thres and candidate > img_height - y_upper_limit:
            bottom = candidate

    if left > x_upper_limit:
        left = 0
    if right < img_width - x_upper_limit:
        right = img_width
    if top > y_upper_limit:
        top = 0
    if bottom < img_height - y_upper_limit:
        bottom = img_height

    return left, right, top, bottom


def find_better_box(box1, box2, box3, box4, multiplier1, multiplier2, multiplier3, multiplier4):
    left = max(box1[0] * multiplier1, box2[0] * multiplier2, box3[0] * multiplier3, box4[0] * multiplier4)
    right = min(box1[1] * multiplier1, box2[1] * multiplier2, box3[1] * multiplier3, box4[1] * multiplier4)
    top = max(box1[2] * multiplier1, box2[2] * multiplier2, box3[2] * multiplier3, box4[2] * multiplier4)
    bottom = min(box1[3] * multiplier1, box2[3] * multiplier2, box3[3] * multiplier3, box4[3] * multiplier4)
    return left, right, top, bottom


# Pyramid
def pyramid(r, g, b, method='edge'):
    all_b_i = 0
    all_b_j = 0

    all_r_i = 0
    all_r_j = 0

    height = r.shape[0]
    pyramid_count = height // 400

    b_aligned = b
    r_aligned = r

    if pyramid_count > 0:
        window_len = pyramid_count + 2
        for power in range(pyramid_count, -1, -1):
            multiplier = power + 1
            b_scale = sk.transform.rescale(b_aligned, 1 / multiplier)
            g_scale = sk.transform.rescale(g, 1 / multiplier)
            r_scale = sk.transform.rescale(r_aligned, 1 / multiplier)
            # print(b_scale.shape)
            crop = int(min(g_scale.shape) * 0.1)

            temp_b_i, temp_b_j = calculate_index(window_len, g_scale, b_scale, crop, method)
            temp_r_i, temp_r_j = calculate_index(window_len, g_scale, r_scale, crop, method)

            # print(temp_b_i, temp_b_j, temp_r_i, temp_r_j)

            b_aligned = np.roll(b_aligned, int(temp_b_i * multiplier), axis=0)
            b_aligned = np.roll(b_aligned, int(temp_b_j * multiplier), axis=1)
            r_aligned = np.roll(r_aligned, int(temp_r_i * multiplier), axis=0)
            r_aligned = np.roll(r_aligned, int(temp_r_j * multiplier), axis=1)

            all_b_i += int(temp_b_i * multiplier)
            all_b_j += int(temp_b_j * multiplier)
            all_r_i += int(temp_r_i * multiplier)
            all_r_j += int(temp_r_j * multiplier)
            # print(all_b_i, all_b_j, all_r_i, all_r_j)

            window_len -= 1
            # crop *= 2
    else:
        window_len = 15
        crop = int(min(g.shape) * 0.1)
        all_b_i, all_b_j = calculate_index(window_len, g, b, crop)
        all_r_i, all_r_j = calculate_index(window_len, g, r, crop)
        b_aligned = np.roll(b, all_b_i, axis=0)
        b_aligned = np.roll(b_aligned, all_b_j, axis=1)
        r_aligned = np.roll(r, all_r_i, axis=0)
        r_aligned = np.roll(r_aligned, all_r_j, axis=1)
    return b_aligned, r_aligned, (all_b_i, all_b_j, all_r_i, all_r_j)


# Auto-cropping

def crop_with_box(img, box):
    left, right, top, bottom = box
    # print(left, right, top, bottom)
    # print(multiplier * top , multiplier * bottom, multiplier * left ,multiplier * right)
    return img[top: bottom, left: right]


# Auto-cropping
def auto_cropping(img, rescale_factor=5):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    im_scale = sk.transform.rescale(img, 1 / rescale_factor)
    r_scale = sk.transform.rescale(r, 1 / rescale_factor)
    g_scale = sk.transform.rescale(g, 1 / rescale_factor)
    b_scale = sk.transform.rescale(b, 1 / rescale_factor)

    img_float32 = np.float32(im_scale)
    r_float32 = np.float32(r_scale)
    g_float32 = np.float32(g_scale)
    b_float32 = np.float32(b_scale)

    gradx = cv2.Sobel(img_float32, cv2.CV_64F, 1, 0, ksize=3)
    grady = cv2.Sobel(img_float32, cv2.CV_64F, 0, 1, ksize=3)

    r_gradx = cv2.Sobel(r_float32, cv2.CV_64F, 1, 0, ksize=3)
    r_grady = cv2.Sobel(r_float32, cv2.CV_64F, 0, 1, ksize=3)

    g_gradx = cv2.Sobel(g_float32, cv2.CV_64F, 1, 0, ksize=3)
    g_grady = cv2.Sobel(g_float32, cv2.CV_64F, 0, 1, ksize=3)

    b_gradx = cv2.Sobel(b_float32, cv2.CV_64F, 1, 0, ksize=3)
    b_grady = cv2.Sobel(b_float32, cv2.CV_64F, 0, 1, ksize=3)

    im_sobel = cv2.magnitude(gradx, grady)
    r_sobel = cv2.magnitude(r_gradx, r_grady)
    g_sobel = cv2.magnitude(g_gradx, g_grady)
    b_sobel = cv2.magnitude(b_gradx, b_grady)

    crop_box_1 = find_final_crop(im_sobel)
    crop_box_2 = find_final_crop(r_sobel)
    crop_box_3 = find_final_crop(g_sobel)
    crop_box_4 = find_final_crop(b_sobel)

    crop_box = find_better_box(crop_box_1, crop_box_2, crop_box_3, crop_box_4, rescale_factor, rescale_factor, rescale_factor, rescale_factor)
    cropped_img = crop_with_box(img, crop_box)

    return cropped_img


# Auto-white balance

def auto_white_balance(img):
    height = int(0.1 * img.shape[0])
    width = int(0.1 * img.shape[1])
    img_scale = cv2.resize(img, (width, height), cv2.INTER_AREA)
    x_interval = int(img_scale.shape[0] * 0.05)
    y_interval = int(img_scale.shape[1] * 0.05)

    x_range = int(img_scale.shape[0] / x_interval)
    y_range = int(img_scale.shape[1] / y_interval)

    # Search white area
    max_area_val = 0
    max_i = 0
    max_j = 0

    pixel = x_interval * y_interval
    for i in range(x_range):
        for j in range(y_range):
            area = img_scale[i * x_interval: (i + 1) * x_interval, j * y_interval: (j + 1) * y_interval]
            if (np.sum(area)) > max_area_val:
                max_area_val = np.sum(area)
                # print(min_area_val)
                max_i = i
                max_j = j

    # Search white area
    min_area_val = 10000
    min_i = 0
    min_j = 0

    pixel = x_interval * y_interval
    for i in range(x_range):
        for j in range(y_range):
            area = img_scale[i * x_interval: (i + 1) * x_interval, j * y_interval: (j + 1) * y_interval]
            if (np.sum(area)) < min_area_val:
                min_area_val = np.sum(area)
                # print(min_area_val)
                min_i = i
                min_j = j

    white_area = img_scale[max_i * x_interval: (max_i + 1) * x_interval, max_j * y_interval: (max_j + 1) * y_interval]
    black_area = img_scale[min_i * x_interval: (min_i + 1) * x_interval, min_j * y_interval: (min_j + 1) * y_interval]

    r_b_area = np.average(black_area[:, :, 0])
    g_b_area = np.average(black_area[:, :, 1])
    b_b_area = np.average(black_area[:, :, 2])

    black_sum = np.sum([r_b_area, g_b_area, b_b_area])

    r_w = 1 - (r_b_area / black_sum) * 0.12
    g_w = 1 - (g_b_area / black_sum) * 0.12
    b_w = 1 - (b_b_area / black_sum) * 0.12

    r_w_area = np.average(white_area[:, :, 0])
    g_w_area = np.average(white_area[:, :, 1])
    b_w_area = np.average(white_area[:, :, 2])

    r_factor = r_w / r_w_area
    g_factor = g_w / g_w_area
    b_factor = b_w / b_w_area

    img_white_balanced = np.dstack([img[:, :, 0] * r_factor, img[:, :, 1] * g_factor, img[:, :, 2] * b_factor])
    return img_white_balanced
