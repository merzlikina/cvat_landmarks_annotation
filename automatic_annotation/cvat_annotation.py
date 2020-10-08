"""
Generates annotation for a given directory with images and a .npy file with
landmarks.
Images filenames must be indices corresponding to vectors in .npy file.
"""
from collections import OrderedDict
from pathlib import Path
import numpy as np
from fire import Fire
from xml.dom import minidom

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("outer_mouth", (48, 60)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def points2str(vec: np.ndarray):
    vec = vec.astype(float)
    vec_str = np.array2string(vec, formatter={'float': '{:0.2f}'.format}).replace('\n ', ';').replace(' ', ',').replace(']', '').replace('[', '')

    return vec_str


def create_polyline_xml(root: minidom.Document, label: str, points: np.ndarray):
    points_str = points2str(points)

    polyline = root.createElement('polyline')

    polyline.setAttribute('label', label)
    polyline.setAttribute('occluded', '0')
    polyline.setAttribute('source', 'dlib')
    polyline.setAttribute('points', points_str)

    return polyline


def create_image_xml(root: minidom.Document, id: int, path: str,
                     points: np.ndarray, image_size: tuple):
    height, width = image_size.astype(int)
    image = root.createElement('image')
    image.setAttribute('id', str(id))
    image.setAttribute('name', path)
    image.setAttribute('height', str(height))
    image.setAttribute('width', str(width))

    for key, (start, end) in FACIAL_LANDMARKS_68_IDXS.items():
        image.appendChild(create_polyline_xml(root, key, points[start:end, :]))

    return image


def get_annotations(landmarks_path: Path):
    root = minidom.Document()
    annotations = root.createElement('annotations')
    root.appendChild(annotations)

    version = root.createElement('version')
    annotations.appendChild(version)

    landmarks = np.load(str(landmarks_path))
    filenames = landmarks['images']
    points = landmarks['points']
    image_sizes = landmarks['image_sizes']

    for ind, (filename, point, image_size) in enumerate(zip(filenames, points, image_sizes)):
        annotations.appendChild(create_image_xml(root, ind, filename, point, image_size))

    with open(landmarks_path.parent / f"{landmarks_path.stem}.xml", 'w') as xml_file:
        root.writexml(xml_file, encoding='utf-8')


def main(path_npy: str):
    """Generate CVAT 1.1 annotations for a provided directory with .npy files
    with images landmarks

    Parameters
    ----------
    path_npy : str
    """
    path_npy = Path(path_npy)
    assert path_npy.is_dir(), "You should provide a directory"

    for npy in path_npy.glob('*.npy'):
        get_annotations(npy)


if __name__ == "__main__":
    Fire(main)
