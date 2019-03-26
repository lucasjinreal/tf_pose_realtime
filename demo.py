import argparse
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from loguru import logger
from alfred.utils.log import init_logger

init_logger()

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='./medias/dance.mp4')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_v2_1.4', help='mobilenet_v2_1.4 cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=True,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()
    
    w, h = 432, 368
    e = TfPoseEstimator('graph/{}/graph_freeze.pb'.format(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    while cap.isOpened():
        ret_val, image = cap.read()
        tic = time.time()
        humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
        if not args.showBG:
            image = np.zeros(image.shape)
        res = TfPoseEstimator.draw_humans(image, humans, imgcopy=True)
        cv2.putText(res, "FPS: %f" % (1.0 / (time.time() - tic)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('rr', res)
        toc = time.time()
        logger.info('inference %.4f seconds.' % (toc-tic))

        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
logger.debug('finished+')
