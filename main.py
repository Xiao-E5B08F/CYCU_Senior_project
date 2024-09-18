import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


# crop_image_list = []


def find_dominant_hand(pose_landmarks):
    center_x = 0.5
    left_wrist = abs(pose_landmarks[15].x - center_x)
    right_wrist = abs(pose_landmarks[16].x - center_x)
    if left_wrist < right_wrist:
        return 15, 21, 19, 17
    else:
        return 16, 22, 20, 18


def unsharp_masking(original_image):
    # import matplotlib.pyplot as plt

    # Convert to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Gaussian filtering
    gray_image_gb = cv2.GaussianBlur(gray_image, (15, 15), 0)

    # Calculate the Laplacian
    lap = cv2.Laplacian(gray_image_gb, cv2.CV_64F)

    # Calculate the sharpened image
    sharp = gray_image - lap

    return sharp


def calculate_roi(pose_landmarks, annotate_image):
    # Ensure the landmarks list has the required landmarks
    wrist, thumb, index, pinky = find_dominant_hand(pose_landmarks)

    # Calculate the distance between the wrist and the thumb finger
    x_diff_top = pose_landmarks[thumb].x - pose_landmarks[wrist].x
    y_diff_top = pose_landmarks[thumb].y - pose_landmarks[wrist].y

    # Calculate the distance between the wrist and the pinky finger
    x_diff_bottom = pose_landmarks[pinky].x - pose_landmarks[wrist].x
    y_diff_bottom = pose_landmarks[pinky].y - pose_landmarks[wrist].y

    # Calculate the extended point
    extend_factor = 15
    diagonal_x_top = pose_landmarks[index].x + x_diff_top * extend_factor
    diagonal_y_top = pose_landmarks[index].y + y_diff_top * extend_factor
    diagonal_x_bottom = pose_landmarks[index].x + x_diff_bottom * extend_factor
    diagonal_y_bottom = pose_landmarks[index].y + y_diff_bottom * extend_factor

    # Calculate the ROI
    height, width, _ = annotate_image.shape
    left_x = int(min(diagonal_x_top, diagonal_x_bottom, pose_landmarks[index].x) * width)
    top_y = int(max(diagonal_y_top, diagonal_y_bottom, pose_landmarks[index].y) * height)
    right_x = int(max(diagonal_x_top, diagonal_x_bottom, pose_landmarks[index].x) * width)
    bottom_y = int(min(diagonal_y_top, diagonal_y_bottom, pose_landmarks[index].y) * height)

    # Crop the image
    crop_image = annotate_image[bottom_y:top_y, left_x:right_x]

    index_x = (pose_landmarks[index].x - left_x / width) / (right_x - left_x)
    index_y = (pose_landmarks[index].y - bottom_y / height) / (top_y - bottom_y)

    return (left_x, top_y), (right_x, bottom_y), crop_image, (index_x, index_y)


def calculate_sword_tip_POSIT(pose_landmarks, annotate_image):
    lt, rb, crop_image, index_finger_pos = calculate_roi(pose_landmarks, annotate_image)
    crop_image_gray = unsharp_masking(crop_image)
    crop_image_gray = np.uint8(crop_image_gray)
    edges = cv2.Canny(crop_image_gray, 30, 60, apertureSize=3)
    lines = cv2.HoughLines(edges, 1.5, np.pi / 180, 80)

    if lines is not None:
        # Iterate through each detected line
        closest_line = None
        closest_distance = float('inf')
        for line in lines:
            rho, theta = line[0]
            # Convert polar coordinates to Cartesian coordinates
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

            # Calculate the distance between the line and the index finger position
            distance = abs((x1 + x2) / 2 - index_finger_pos[0]) + abs((y1 + y2) / 2 - index_finger_pos[1])

            # Update the closest line if the distance is smaller
            if distance < closest_distance:
                closest_line = (x1, y1, x2, y2)
                closest_distance = distance

        # Draw the closest line on the crop image
        if closest_line is not None:
            cv2.line(crop_image, (closest_line[0], closest_line[1]), (closest_line[2], closest_line[3]), (0, 0, 255), 2)

    # crop_image_list.append(edges)
    return lt, rb


def draw_sword_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotate_image = np.copy(rgb_image)
    rect_list = []
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]  # The ith person's pose landmarks

        # Draw the sword tip point
        index1, index2 = calculate_sword_tip_POSIT(pose_landmarks, annotate_image)
        rect_list.append(index1)
        rect_list.append(index2)

    # Draw the rectangle
    '''
    if len(rect_list) == 4:
        cv2.rectangle(annotate_image, rect_list[0], rect_list[1], (0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.rectangle(annotate_image, rect_list[2], rect_list[3], (0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    '''
    return annotate_image


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotate_image = np.copy(rgb_image)
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]  # The ith person's pose landmarks

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotate_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())

    return annotate_image


def create_pose_landmarker(filename, model_asset_path):
    # Create an PoseLandmarker object.
    # setting what kind of model we use
    base_options = mp.tasks.BaseOptions(model_asset_path=model_asset_path)
    #                                    , delegate=mp.tasks.BaseOptions.Delegate.GPU)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2)  # setting options

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        # Load the input video and setting output video.
        video = cv2.VideoCapture("./Data/" + filename)
        if not video.isOpened():
            print("Cannot open camera")
            exit()
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter("./Output/Drew_" + filename, fourcc, fps, (width, height))

        percent_target = 0
        # Loop every frame and annotated the frame
        total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(int(total_frame)):
            success, frame = video.read()
            if not success:
                break
            # frame = cv2.resize(frame, (width, height))  # resize
            frame_timestamp_ms = int(video.get(cv2.CAP_PROP_POS_MSEC))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)  # detect the pose
            if detection_result.pose_landmarks is None:
                cv2.imshow("aa", frame)
                cv2.waitKey(0)
                cv2.destroyWindow('aa')
                output.write(frame)
            else:
                try:
                    annotated_image = draw_sword_on_image(frame, detection_result)
                    annotated_image = draw_landmarks_on_image(annotated_image,
                                                              detection_result)  # draw the landmarks on the frame
                except:
                    pass

                output.write(annotated_image)
            if int(100 * i / total_frame) == percent_target:
                print("{:.2f}%".format(int(100 * i / total_frame)))
                percent_target += 10
        video.release()
        output.release()
        cv2.destroyAllWindows()


def main():
    filename = "p4.mp4"
    model_asset_path = 'pose_landmarker_full.task'
    create_pose_landmarker(filename, model_asset_path)

    '''
    i = 0
    c = 1
    for frame in crop_image_list:
        i += 1
        if i % 30 == 0:
            cv2.imwrite("./sword_detect pic/crop_image" + str(c) + ".jpg", frame)
            i = 0
            c = c + 1
    '''


if __name__ == '__main__':
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(total_time)
