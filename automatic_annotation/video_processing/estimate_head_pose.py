"""This script is a modified version of estimate_head_pose.py from
https://github.com/yinguobing/head-pose-estimation
"""
import cv2
import numpy as np
from pathlib import Path

from head_pose_estimation.mark_detector import MarkDetector
from head_pose_estimation.pose_estimator import PoseEstimator
from head_pose_estimation.stabilizer import Stabilizer
import fire

print("OpenCV version: {}".format(cv2.__version__))

CNN_INPUT_SIZE = 128


def euler_angles(pose):
    # Solve the pitch, yaw and roll angels.
    r_mat, _ = cv2.Rodrigues(pose[0])
    p_mat = np.hstack((r_mat, np.array([[0], [0], [0]])))
    _, _, _, _, _, _, u_angle = cv2.decomposeProjectionMatrix(p_mat)
    pitch, yaw, roll = u_angle.flatten()

    # I do not know why the roll axis seems flipted 180 degree. Manually by pass
    # this issue.
    if roll > 0:
        roll = 180-roll
    elif roll < 0:
        roll = -(180 + roll)

    return np.array([pitch, yaw, roll])


def save_landmarks(filename: str, indices: np.ndarray, points: np.ndarray, head_poses: np.ndarray, facebox: np.ndarray):
    n, num_rows, num_col = points.shape

    wtype = np.dtype([('indices', indices.dtype), ('points', points.dtype, (num_rows, num_col)), ('head_poses', head_poses.dtype, (3,)), ('facebox', facebox.dtype, (4,))])
    w = np.empty(n, dtype=wtype)

    w['indices'] = indices
    w['points'] = points
    w['head_poses'] = head_poses
    w['facebox'] = facebox

    # TODO: debug facebox behavior
    print(facebox)
    np.save(filename, w)


def process_video(path):
    landmarks = []
    head_poses = []
    indices = []
    faceboxes = []

    cap = cv2.VideoCapture(path)

    _, sample_frame = cap.read()
    i = 1
    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        facebox = mark_detector.extract_cnn_facebox(frame)
        if (facebox is not None) and (len(mark_detector.face_detector.detection_result[0]) == 1):
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            tm.start()
            marks = mark_detector.detect_marks([face_img])
            tm.stop()

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # Uncomment following line to show raw marks.
            mark_detector.draw_marks(
                frame, marks, color=(0, 255, 0))

            # Uncomment following line to show facebox.
            # mark_detector.draw_box(frame, [facebox])

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))

            pose = euler_angles(steady_pose)

            landmarks.append(marks)
            head_poses.append(pose)
            indices.append(i)
            faceboxes.append(facebox)

        i += 1

    landmarks = np.array(landmarks)
    head_poses = np.array(head_poses)
    indices = np.array(indices)
    faceboxes = np.array(faceboxes)

    return indices, landmarks, head_poses, faceboxes


def main(path: str):
    path = Path(path)

    files = [p.resolve() for p in Path(path).glob("*") if p.suffix in [".mp4", ".webm", ".avi"]]

    dir_landmarks = path / "npy_landmarks"
    dir_landmarks.mkdir(exist_ok=True)

    for i in files:
        # TODO: add logger
        print(f"Processing {i}")

        if (dir_landmarks/f"{i.stem}.npy").is_file():
            continue
        indices, landmarks, head_poses, faceboxes = process_video(str(i))

        if len(indices) > 0:
            save_landmarks(str(dir_landmarks/f"{i.stem}.npy"), indices, landmarks, head_poses, faceboxes)


if __name__ == '__main__':
    fire.Fire(main)
