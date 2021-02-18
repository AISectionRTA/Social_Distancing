import os
import cv2
import time
import argparse
import numpy as np
from PIL import Image
from Social_Distancing.network_model import model
from Social_Distancing.aux_functions import *
from Social_Distancing.utils.anchor_generator import generate_anchors
from Social_Distancing.utils.anchor_decode import decode_bbox
from Social_Distancing.utils.nms import single_class_non_max_suppression
from Social_Distancing.load_model.tensorflow_loader import load_tf_model, tf_inference
from scipy.ndimage import zoom

# "Lift_videos/Lift.mp4"
# "Bus_Videos/_ (1).mp4"
default = "Lift_videos/Lift.mp4"
scale_w = 1.0
scale_h = 1.0

warning = 0
violations_counter = 1
# Suppress TF warnings
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

mouse_pts = []

sess, graph = load_tf_model('models/face_mask_detection.pb')
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
msk_index = 0


def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    # image = np.copy(image)
    global msk_index
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0
    image_exp = np.expand_dims(image_np, axis=0)
    y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)

    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
                msk_index = msk_index + 1
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
    return output_info


def run_on_video(video_path, output_video_name, conf_thresh):
    status = True
    while status:
        status, img_raw = cap.read()
        img_raw = cv2.resize(img_raw, (0, 0), fx=2.3, fy=1.3)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        if (status):
            inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(260, 260),
                      draw_result=True,
                      show_result=False)
            cv2.imshow('', img_raw[:, :, ::-1])


def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 6 feet away
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 10, (0, 255, 255), 10)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


# Command-line input setup
parser = argparse.ArgumentParser(description="SocialDistancing")
parser.add_argument(
    "--videopath", type=str, default=default, help="Path to the video file"
)
args = parser.parse_args()

input_video = args.videopath

# Define a DNN model
DNN = model()
# Get video handle
cap = cv2.VideoCapture(input_video)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_h)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_w)
fps = int(cap.get(cv2.CAP_PROP_FPS))

SOLID_BACK_COLOR = (41, 41, 41)
# Setuo video writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_movie = cv2.VideoWriter("SocialDistancing_detect.avi", fourcc, fps, (width, height))
bird_movie = cv2.VideoWriter(
    "SocialDistancing_bird.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h))
)
# Initialize necessary variables
frame_num = 0
total_pedestrians_detected = 0
total_six_feet_violations = 0
total_pairs = 0
abs_six_feet_violations = 0
pedestrian_per_sec = 0
sh_index = 1
sc_index = 1

cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
num_mouse_points = 0
first_frame_display = True

# Process each frame, until end of video
while cap.isOpened():
    frame_num += 1
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (0, 0), fx=scale_w, fy=scale_h)
    # frame = clipped_zoom(frame, 0.5)
    frame = cv2.copyMakeBorder( frame, 100, 100, 100, 100, cv2.BORDER_CONSTANT)

    if not ret:
        print("end of the video file...")
        break

    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    if frame_num == 1:
        # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2
        while True:
            image = frame
            cv2.imshow("image", image)
            cv2.waitKey(1)
            if len(mouse_pts) == 7:
                cv2.destroyWindow("image")
                break
            first_frame_display = False
        four_points = mouse_pts

        # Get perspective
        M, Minv = get_camera_perspective(frame, four_points[0:4])
        pts = src = np.float32(np.array([four_points[4:]]))
        warped_pt = cv2.perspectiveTransform(pts, M)[0]
        d_thresh = np.sqrt(
            (warped_pt[0][0] - warped_pt[1][0]) ** 2
            + (warped_pt[0][1] - warped_pt[1][1]) ** 2
        )
        bird_image = np.zeros(
            (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
        )

        bird_image[:] = SOLID_BACK_COLOR
        pedestrian_detect = frame

    # print("Processing frame: ", frame_num)
    print('Warning = ' + str(warning))

    # draw polygon of ROI
    pts = np.array(
        [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
    )
    cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=4)

    # Detect person and bounding boxes using DNN
    pedestrian_boxes, num_pedestrians = DNN.detect_pedestrians(frame)

    if len(pedestrian_boxes) > 0:
        pedestrian_detect = plot_pedestrian_boxes_on_image(frame, pedestrian_boxes)
        warped_pts, bird_image = plot_points_on_bird_eye_view(
            frame, pedestrian_boxes, M, scale_w, scale_h
        )
        six_feet_violations, ten_feet_violations, pairs = plot_lines_between_nodes(
            warped_pts, bird_image, d_thresh
        )
        # plot_violation_rectangles(pedestrian_boxes, )
        total_pedestrians_detected += num_pedestrians
        total_pairs += pairs

        total_six_feet_violations += six_feet_violations / fps
        abs_six_feet_violations += six_feet_violations
        pedestrian_per_sec, sh_index = calculate_stay_at_home_index(
            total_pedestrians_detected, frame_num, fps
        )
        if six_feet_violations > 0:
            warning = warning + six_feet_violations
    if 1 < warning < 10:
        text = "Distancing violations"  # + str(int(total_six_feet_violations))
        pedestrian_detect, last_h = put_text(pedestrian_detect, text)
    elif warning > 10:
        text = "Danger Distancing violations "  # + str(int(total_six_feet_violations))
        pedestrian_detect, last_h = put_text(pedestrian_detect, text)
        cv2.imwrite('Violation/violation_Distancing_' + str(violations_counter) + '.jpg', frame)
        violations_counter = violations_counter + 1
    if warning > 0:
        warning = warning - 1

    # text = "Mask violations: " + str(msk_index)
    # pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)
    # sh_index
    # text = "Stay-at-home Index: " + str(np.round(100 * sh_index, 1)) + "%"
    # pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

    # if total_pairs != 0:
    # sc_index = 1 - abs_six_feet_violations / total_pairs

    ##text = "Safety Index: " + str(np.round(100 * sc_index, 1)) + "%"
    # pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

    """
    img_raw = cv2.cvtColor(pedestrian_detect, cv2.COLOR_BGR2RGB)
    inference(img_raw,
              conf_thresh=0.9,
              iou_thresh=0.9,
              target_shape=(260, 260),
              draw_result=True,
              show_result=False)
    cv2.imshow("Street Cam", img_raw[:, :, ::-1])
    cv2.waitKey(1)
    output_movie.write(img_raw[:, :, ::-1])
    """
    cv2.imshow("Street Cam", pedestrian_detect)
    cv2.waitKey(1)
    output_movie.write(pedestrian_detect)

    bird_movie.write(bird_image)
