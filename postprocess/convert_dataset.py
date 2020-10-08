import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import os
import numpy as np
import fire

from .xml_template import create_xml

# create directory for zip unpacking in the root of the project
path_annotated = (Path.cwd() / Path(__file__)).parent.parent / 'annotated'
path_annotated.mkdir(exist_ok=True)

facial_parts = ['jaw', 'right_eyebrow', 'left_eyebrow', 'nose', 'right_eye',
                'left_eye', 'outer_mouth', 'inner_mouth']


def str2array(string: str):
    return np.fromstring(string.replace(';', ','), sep=',').reshape(-1, 2)


def convert_points(points: dict):
    all_points = ','.join([points.get(key) for key in facial_parts])
    return str2array(all_points)


def parse_cvat_annotation(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    points = []
    filenames = []

    for image in root.findall('image'):
        filenames.append(image.attrib['name'])

        image_element = {}
        for polyline in image:
            image_element[polyline.attrib['label']] = polyline.attrib['points']
        points.append(convert_points(image_element))

    return filenames, points


def expand_filenames(filenames: list, zip_filename: str) -> list:
    """Expand image filenames with directory path
    """
    processed = [os.path.join(zip_filename, 'images', i) for i in filenames]

    return processed


def process_zip(path_zip: str):
    path_zip = Path(path_zip)
    zip_filename = path_zip.stem

    res_dir = path_annotated / zip_filename
    res_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(str(res_dir))

    try:
        filenames, points = parse_cvat_annotation(str(res_dir / 'annotations.xml'))
        filenames = expand_filenames(filenames, zip_filename)
    except Exception as e:
        print(e)
        return None
    else:
        return filenames, points


def read_annotated(path: str):
    """Find all

    Parameters
    ----------
    path : str
        Path to a directory with downloaded from cvat zip files
    """
    assert Path(path).is_dir(), "Provided path must be a directory"

    points_all = []
    filenames_all = []

    for zip_file in Path(path).glob("*-cvat for images 1.1.zip"):
        processed = process_zip(zip_file)
        if processed is not None:
            filenames, points = processed

            points_all.append(points)
            filenames_all.append(filenames)
    points_all = np.array(points_all)
    n_zip, n_entries, n_points, n_dims = points_all.shape
    points_all = points_all.reshape(-1, n_points, n_dims).astype(int)

    filenames_all = [item for sublist in filenames_all for item in sublist]
    return points_all, filenames_all


def main(path_dir_zip: str):
    assert Path(path_dir_zip).is_dir(), "Provided path must be a directory with \
                                     downloaded annotation .zip files"

    points_all, filenames_all = read_annotated(path_dir_zip)
    create_xml(filenames_all, points_all)


if __name__ == "__main__":
    fire.Fire(main)
