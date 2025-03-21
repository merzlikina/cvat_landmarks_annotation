import numpy as np
from pathlib import Path
import cv2
import fire
from .estimate_head_pose import process_video
ALLOWED_DEVIATION = 35
CUMMULATIVE_CHANGE = 15


def sin_cos(x):
    return np.sin(x), np.cos(x)


def from_euler_angles(roll, pitch, yaw):
    (sr, cr) = sin_cos(roll)
    (sp, cp) = sin_cos(pitch)
    (sy, cy) = sin_cos(yaw)

    return [[cy * cp,
             cy * sp * sr - sy * cr,
             cy * sp * cr + sy * sr],
            [sy * cp,
             sy * sp * sr + cy * cr,
             sy * sp * cr - cy * sr],
            [-sp,
             cp * sr,
             cp * cr]]


def to_angle(m: np.ndarray) -> float:
    return np.arccos((np.trace(m)-1)/2)


def measure_deviation(angles):
    """Measure deviation of head position from still position
    (to later discard strong head rotations)
    """
    deviations = []
    zero_inv = np.linalg.inv(from_euler_angles(0, 0, 0))
    for angle in angles:
        deviations.append(np.rad2deg(
            to_angle(from_euler_angles(*angle) @ zero_inv)))

    return np.array(deviations)


def diff_points(x1, x2):
    return np.rad2deg(np.abs(np.sum(x1-x2)))


def diff(arr, point, ind):
    for i, x in enumerate(arr):
        if diff_points(x, point) > CUMMULATIVE_CHANGE:
            return i + ind
    return ind


def get_significant_indices(angles, indices):
    flag = False
    significant_ind = []
    significant_ind.append(indices[0])
    temp_ind = 0

    while not flag:
        ind = diff(angles[temp_ind:], angles[temp_ind], temp_ind)
        if ind != temp_ind:
            temp_ind = ind
            significant_ind.append(indices[ind])
        else:
            flag = True
    return significant_ind


def read_landmarks(filename):
    data = np.load(filename)

    return data['indices'], data['points'], data['head_poses'], data['image_size']


def extract_frames(filename):
    path_images = Path(filename).parent / "images"
    path_images.mkdir(exist_ok=True)

    indices, landmarks, head_poses = process_video(str(filename))

    if len(indices) > 0:
        path_images_videofile = path_images / Path(filename).stem
        path_images_videofile.mkdir(exist_ok=True)

        angles = np.deg2rad(head_poses)
        # discard head positions where head is "too" rotated
        mask = measure_deviation(angles) < ALLOWED_DEVIATION
        indices_ = indices[mask]
        angles_ = angles[mask]
        # all headposes vere too extreme
        if len(indices_) == 0:
            return
        significant_ind = get_significant_indices(angles_, indices_)
        # deal with opencv issue with reading end of webm video
        if Path(filename).suffix == '.webm':
            cap = cv2.VideoCapture(str(filename))
            i = 0
            while cap.isOpened() and i <= max(significant_ind):
                ret, frame = cap.read()
                if not ret:
                    break
                if i in significant_ind:
                    cv2.imwrite(str(path_images_videofile/f"{i}.jpg"), frame)
                i += 1
            cap.release()
        else:
            for i in significant_ind:
                cap = cv2.VideoCapture(str(filename))
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                _, frame = cap.read()
                cv2.imwrite(str(path_images_videofile/f"{i}.jpg"), frame)
                cap.release()


def main(path: str):
    """Extract distinguished frames from a provided directory with video files.

    Parameters
    ----------
    path : str
        Path to a directory with video files
    """
    # deal with bash "\ " inserting for paths with spaces
    path = Path(path.replace("\ ", " "))
    assert path.is_dir(), "Provided path must be a directory"

    files = [p.resolve() for p in Path(path).glob(
        "*") if p.suffix in [".mp4", ".webm", ".avi"]]
    for filename in files:
        # TODO: add logger
        print(f"Processing {filename}")
        extract_frames(str(filename))


if __name__ == "__main__":
    fire.Fire(main)
