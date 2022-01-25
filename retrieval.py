import cv2
import numpy as np
from skimage.io import imread, imshow
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_template
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

def upgrade_predict(img: np.ndarray, posarr ):
    img = img
    img_gray = rgb2gray(img)
    template = img_gray[posarr[1]:posarr[3],posarr[0]:posarr[2]]
    resulting_image = match_template(img_gray, template)
    x, y = np.unravel_index(np.argmax(resulting_image), resulting_image.shape)
    template_width, template_height = template.shape
    img_width, img_height = img_gray.shape
    #plt.figure(num=None, figsize=(8, 6), dpi=80)
    list_of_bboxes =[]
    for x, y in peak_local_max(resulting_image,min_distance = 30, threshold_abs=0.45,threshold_rel=0.4,
                               exclude_border=0):
        #rect = plt.Rectangle((y, x), template_height, template_width,
        #                   color='r', fc='none')
        list_of_bboxes.append((x/img_width, y/img_height,(x+template_width)/img_width,(y+template_height)/img_height))
        #plt.gca().add_patch(rect)
    #imshow(img)
    #plt.show()
    return list_of_bboxes

def predict_image(img: np.ndarray, query: np.ndarray) -> list:
    MIN_MATCH_COUNT = 6
    DIST_COEFF = 0.75
    MIN_MATCH_COUNT1 = 8
    DIST_COEFF1 = 0.7
    list_of_bboxes = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)#cv2.BFMatcher()
    kp_big, des_big = sift.detectAndCompute(img_gray, None)
    kp_tpl, des_tpl = sift.detectAndCompute(query, None)
    matches = matcher.knnMatch(des_tpl, des_big, k=2)
    good = []
    img_height,img_width  = img_gray.shape
    for m, n in matches:
        if m.distance < n.distance * DIST_COEFF1:
            good.append(m)
    if len(good) < MIN_MATCH_COUNT1:
        pass
    else:
        while True:
            src_pts = np.float32([kp_tpl[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_big[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = query.shape
            pts = np.asarray([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype=np.float32).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            dst = [np.int32(np.abs(dst))]
            img_res = img
            #print(dst)
            #print(dst[0][0][0])
            top_left = dst[0][0][0]
            #print(dst[0][2][0])
            bot_right = dst[0][2][0]

            img_res = cv2.rectangle(img_res, top_left,bot_right,(0, 255, 255), 2)
            rect = [dst[0][0][0],dst[0][1][0],dst[0][2][0],dst[0][3][0]]
            img_gray = cv2.fillPoly(img_gray, dst, 0)
            crop = img[top_left[1]+1:bot_right[1]+1, top_left[0]+1:bot_right[0]+1]
            try:
                kp_big, des_big = sift.detectAndCompute(img_gray, None)
                kp_tpl, des_tpl = sift.detectAndCompute(crop, None)
            except:
                break
            list_of_bboxes.append((top_left[0] / img_width, top_left[0] / img_height, (bot_right[0]) / img_width,
                                   (bot_right[1]) / img_height))
            matches = matcher.knnMatch(des_tpl, des_big, k=2)
            good = []
            # resize = cv2.resize(img_gray, (600, 600), interpolation=cv2.INTER_AREA)
            #
            # cv2.imshow('sda', resize)
            # cv2.waitKey(0)
            for m, n in matches:
                if m.distance < n.distance * DIST_COEFF:
                    good.append(m)
            if len(good) < MIN_MATCH_COUNT:
                break
            #list_of_bboxes = upgrade_predict(img, (top_left[0], top_left[1], bot_right[0], bot_right[1]))

    #resize = cv2.resize(img_res, (600, 600), interpolation=cv2.INTER_AREA)

    #cv2.imshow('sda',resize)
    #cv2.waitKey(0)
    #list_of_bboxes = [(0, 0, 1, 1), ]
    return list_of_bboxes

# img = cv2.imread('Data/train_extreme.jpg')
# # #
# query = cv2.imread('Data/template_extreme.jpg')
# bboxes_list= predict_image(img,query)
# print(bboxes_list)
# print(len(bboxes_list))

