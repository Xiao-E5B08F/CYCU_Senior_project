import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

#TODO The index shouldn't be hardcoded and should consider the dominant hand.
def calculate_extended_point(pose_landmarks, index1 = 16, index2 = 20, multiplier = 10):
    # Ensure the landmarks list has the required landmarks
    if len(pose_landmarks) > max(index1, index2):
        x_diff = pose_landmarks[index2].x - pose_landmarks[index1].x
        y_diff = pose_landmarks[index2].y - pose_landmarks[index1].y

        # Calculate the new point
        new_x = pose_landmarks[index2].x + x_diff * multiplier
        new_y = pose_landmarks[index2].y + y_diff * multiplier

        return new_x, new_y
    else:
        print(f"Landmarks list does not have the required landmarks: {index1}, {index2}")
        return None, None


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotate_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Calculate the extended point
        new_x, new_y = calculate_extended_point(pose_landmarks)
        if new_x is not None and new_y is not None:
            # Convert the normalized coordinates to pixel coordinates
            height, width, _ = rgb_image.shape
            pixel_x = int(new_x * width)
            pixel_y = int(new_y * height)

            # Draw the extended point on the image
            cv2.circle(annotate_image, (pixel_x, pixel_y), radius=5, color=(0, 255, 0), thickness=-1)

            # Draw a line from index2 to the extended point
            #TODO The index shouldn't be hardcoded.
            index2_x = int(pose_landmarks[20].x * width)
            index2_y = int(pose_landmarks[20].y * height)
            cv2.line(annotate_image, (index2_x, index2_y), (pixel_x, pixel_y), color=(0, 255, 0), thickness=2)
        

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
    base_options = mp.tasks.BaseOptions(model_asset_path='pose_landmarker_full.task')
    #                                    , delegate=mp.tasks.BaseOptions.Delegate.GPU)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2)  # setting options

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        # Load the input video and setting output video.
        filename = "Fencing1080p.mp4"
        video = cv2.VideoCapture(filename)
        if not video.isOpened():
            print("Cannot open camera")
            exit()
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter("Drew_" + filename, fourcc, fps, (width, height))

        percent_target = 0
        # Loop every frame and annotated the frame
        total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(int(total_frame)):
            success, frame = video.read()
            # frame = cv2.resize(frame, (width, height))  # resize
            frame_timestamp_ms = int(video.get(cv2.CAP_PROP_POS_MSEC))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)  # detect the pose
            annotated_image = draw_landmarks_on_image(frame, detection_result)  # draw the landmarks on the frame
            output.write(annotated_image)
            '''
            cv2.imshow("annotated_image", annotated_image)
            if cv2.waitKey(1) == ord('q'):
                break
            '''
            if int(100 * i / total_frame) == percent_target:
                print("{:.2f}%".format(int(100 * i / total_frame)))
                percent_target += 10
        video.release()
        output.release()
        cv2.destroyAllWindows()


def main():
    create_pose_landmarker()


if __name__ == '__main__':
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(total_time)
    