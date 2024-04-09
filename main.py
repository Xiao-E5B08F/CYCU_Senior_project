import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def draw_landmarks_on_image(rgb_image, result):
    pose_landmarks_list = result.pose_landmarks
    # pose_landmarks_list = detection_result
    annotate_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

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


def create_pose_landmarker():
    # Create an PoseLandmarker object.
    # setting what kind of model we use
    base_options = mp.tasks.BaseOptions(model_asset_path='pose_landmarker.task')
    # ,delegate=python.BaseOptions.Delegate.GPU)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2)  # setting options

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        # Load the input video and setting output video.
        video = cv2.VideoCapture("Fencing.mp4")
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter("Fencing_drew.mp4", fourcc, fps, (width, height))

        percent_target = 1
        # Loop every frame and annotated the frame
        total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(int(total_frame)):
            success, frame = video.read()
            frame = cv2.resize(frame, (520, 300))  # resize
            frame_timestamp_ms = int(video.get(cv2.CAP_PROP_POS_MSEC))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            annotated_image = draw_landmarks_on_image(frame, detection_result)
            output.write(annotated_image)
            # cv2.imshow('frame', annotated_image)

            if int(100 * i / total_frame) == percent_target:
                print("{:.2f}%".format(int(100 * i / total_frame)))
                percent_target += 1


def main():
    create_pose_landmarker()


if __name__ == '__main__':
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(total_time)
