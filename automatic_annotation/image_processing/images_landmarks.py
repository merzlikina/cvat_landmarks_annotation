import numpy as np
import dlib
import cv2
from configparser import ConfigParser
from pathlib import Path
import logging
from random import shuffle
from shutil import copyfile
import os
import fire

logging.basicConfig(level=logging.INFO)

config = ConfigParser()
config.read('../config.ini')

path_shape_predictor = str(Path(config['PATHS']["SHAPE_PREDICTOR_68_PATH"]) /
                           config['PATHS']["SHAPE_PREDICTOR_68"])

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_shape_predictor)


def save_landmarks(dir_images: str, image_filenames: np.ndarray, points: np.ndarray):
    dir_images = Path(dir_images)
    n, num_rows, num_col = points.shape
    # points = points.reshape(-1, num_rows*num_col)

    wtype = np.dtype([('images', image_filenames.dtype),
                      ('points', points.dtype, (num_rows, num_col))])
    w = np.empty(len(image_filenames), dtype=wtype)
    w['images'] = image_filenames
    w['points'] = points

    np.save(dir_images.parent/f'landmarks_{dir_images.parts[-1]}.npy', w)


def dlib2rect(rect):
    """converts dlib rectangle coordinates to opencv

        Parameters
        ----------
        rect
            dlib rectangle object
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def shape2np(shape):
    return np.array([[p.x, p.y] for p in shape.parts()])


class Image:
    def __init__(self, path_image):
        # load the input image, resize it, and convert it to grayscale
        self.image = cv2.imread(str(path_image))
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.rect = self.detect_face(self.gray)

        if self.rect is not None:
            self.points = self.detect_landmarks(self.gray, self.rect)

    @staticmethod
    def detect_face(gray: np.ndarray):
        """Find face rect on a given image
        """

        # detect faces in the grayscale image
        rects_dlib = detector(gray, 1)

        if len(rects_dlib) > 0:
            # convert rects to (x, y, w, h)
            rects = [dlib2rect(i) for i in rects_dlib]

            # extract widths and find index of the biggest rect
            widths = [rect[2] for rect in rects]

            ind_max = np.argmax(widths)

            return rects_dlib[ind_max]
        else:
            return None

    @staticmethod
    def detect_landmarks(gray: np.ndarray, rect):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        return shape2np(shape)


def detect_face(gray: np.ndarray):
    """Find face rect on a given image
    """

    # detect faces in the grayscale image
    rects_dlib = detector(gray, 1)
    if len(rects_dlib) > 0:
        # convert rects to (x, y, w, h)
        rects = [dlib2rect(i) for i in rects_dlib]

        # extract widths and find index of the biggest rect
        widths = [rect[2] for rect in rects]

        ind_max = np.argmax(widths)

        return rects_dlib[ind_max]
    else:
        return None


def detect_landmarks(gray: np.ndarray, rect):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    return shape2np(shape)


def process_images(dir_images: str):
    LIMIT = 100
    files = [p.resolve() for p in Path(dir_images).rglob("*") if p.suffix in [".png", ".jpg"]]
    shuffle(files)

    N = len(files) // LIMIT
    if N == 0:
        N = 1
    logging.info(f'Detected {len(files)} files.')
    logging.info(f'Splitting into {N} chunks.')

    # Path to root of the project
    data_path = (Path(os.getcwd()) / Path(__file__)).parent.parent / 'data'
    data_path.mkdir(exist_ok=True)

    chunks_path = data_path / (Path(dir_images).parts[-1] + "_chunks")
    chunks_path.mkdir(exist_ok=True)

    for i, part in enumerate(np.array_split(np.array(files), N)):
        filenames = []
        all_points = []

        chunk_dir = Path(chunks_path / f"{i}")
        chunk_dir.mkdir(exist_ok=True)

        for path_image in part:
            img = Image(str(path_image))

            # check if bounding face rect is inside picture
            if img.rect is not None:
                filenames.append(str(path_image.parts[-1]))
                all_points.append(img.points)
                copyfile(path_image, str(chunk_dir / path_image.parts[-1]))

        logging.info(f"Chunk {i} was processed.")

        save_landmarks(f"{chunks_path}_{i}", np.array(filenames), np.array(all_points))


if __name__ == "__main__":
    process_images() # provide directory
