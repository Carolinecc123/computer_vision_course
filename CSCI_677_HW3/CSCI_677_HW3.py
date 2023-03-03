import cv2 as cv
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def q1():
    img_dir = './HW3_Data/'

    img_list = os.listdir(img_dir)
    for img_path in img_list:
        img_id = img_path[:-4]
        img_name = os.path.join(img_dir, img_path)

        img = cv.imread(img_name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp = sift.detect(gray, None)
        img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite( 'sift/' + img_id + '_sift_keypoints.jpg', img)
        print(f'The number of features detected in {img_id}.jpg is {len(kp)}')

def q2(args):
    img_dir = './HW3_Data/'
    img_list = os.listdir(img_dir)
    src_des = []
    src_imgs = []
    src_kp = []
    dst_des = []
    dst_imgs = []
    dst_kp = []
    # initial SIFT
    sift = cv.SIFT_create()
    # Calculate keypoints and description
    # Store them in list
    for img_path in img_list:
        img_id = img_path[:-4]
        img_name = os.path.join(img_dir, img_path)

        img = cv.imread(img_name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if img_id[:3] == 'src':
            src_des.append(des)
            src_imgs.append(img)
            src_kp.append(kp)
        else:
            dst_des.append(des)
            dst_imgs.append(img)
            dst_kp.append(kp)

    # For each pair use Brute force to match
    bf = cv.BFMatcher()
    for i in range(len(src_imgs)):
        for j in range(len(dst_imgs)):
            matches = bf.knnMatch(src_des[i], dst_des[j], k=args.k)

            # Apply ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            good = sorted(good, key=lambda x: x[0].distance)
            # cv.drawMatchesKnn expects list of lists as matches.
            img = cv.drawMatchesKnn(src_imgs[i], src_kp[i], dst_imgs[j], dst_kp[j], good[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imwrite(f'match/src_{i}_dst_{j}.jpg', img)
            print(f'The number of match pairs for src_{i} and dst_{j} is {len(good)}')

def q34(args):
    img_dir = './HW3_Data/'
    img_list = os.listdir(img_dir)
    src_des = []
    src_imgs = []
    src_kp = []
    dst_des = []
    dst_imgs = []
    dst_kp = []
    # initial SIFT
    sift = cv.SIFT_create()

    # Calculate keypoints and description
    # Store them in list
    for img_path in img_list:
        img_id = img_path[:-4]
        img_name = os.path.join(img_dir, img_path)

        img = cv.imread(img_name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if img_id[:3] == 'src':
            src_des.append(des)
            src_imgs.append(img)
            src_kp.append(kp)
        else:
            dst_des.append(des)
            dst_imgs.append(img)
            dst_kp.append(kp)


    bf = cv.BFMatcher()
    for i in range(len(src_imgs)):
        for j in range(len(dst_imgs)):

            # Brute force match
            matches = bf.knnMatch(src_des[i], dst_des[j], k=args.k)

            # Apply ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) > args.MIN_MATCH_COUNT:
                src_pts = np.float32([src_kp[i][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([dst_kp[j][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()
                h, w, d = src_imgs[i].shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, M)
                img = dst_imgs[j].copy()
                img = cv.polylines(img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            else:
                print("Not enough matches are found - {}/{}".format(len(good), args.MIN_MATCH_COUNT))
                matchesMask = None


            # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
            #                    singlePointColor=None,
            #                    matchesMask=matchesMask,  # draw only inliers
            #                    flags=2)
            # res_img = cv.drawMatches(src_imgs[i], src_kp[i], img, dst_kp[j], good, None, **draw_params)

            res_img, average_dst = draw_top10(src_imgs[i], src_kp[i], img, dst_kp[j], good, M)

            cv.imwrite(f'inliner_matches/src_{i}_dst_{j}.jpg', res_img)


            # Calculate the number of inliner match pairs
            # print out the answer for q3 and q4
            # print(f'The number of inliner match pairs for src_{i} and dst_{j} is {matchesMask.count(1)}')
            # print(f'The computed homography matrix for src_{i} and dst_{j}:')
            # print(M)
            print(f'The average of top 10 minial error for src_{i} and dst_{j}: {average_dst}')


def draw_top10(src_img, src_kp, dst_img, dst_kp, good, M):
    """
    @func: draw top-10 matches that have the minimum error between
           the projected source keypoint and the destination keypoint
    """
    src_pts = np.float32([src_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # project source points
    prj_src_pts = cv.perspectiveTransform(src_pts, M)
    # calculate l2 distance
    l2_distance = np.sqrt(np.square(dst_pts-prj_src_pts).sum(axis=2))

    # sort distances between projected source points and dsetination points
    idxs = np.argsort(l2_distance.squeeze(), axis=0).tolist()[:10]

    # set top 10 to 1
    matchesMask = [0] * src_pts.shape[0]
    for idx in idxs:
        matchesMask[idx] = 1


    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    res_img = cv.drawMatches(src_img, src_kp, dst_img, dst_kp, good, None, **draw_params)

    top_10_dst = l2_distance[idxs]
    average_dst = top_10_dst.mean()

    return res_img, average_dst



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--MIN_MATCH_COUNT', type=int, default=10)
    args =parser.parse_args()
    # q1()
    # q2(args)
    q34(args)