from xml.dom import minidom
import numpy as np

RESULT_XML = 'result.xml'


def get_bbox(points: np.ndarray):
    """Estimate face bounding box from facial landmarks

    Parameters
    ----------
    points : np.ndarray
        [description]
    """
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    width = x_max - x_min
    height = y_max - y_min

    length = max(width, height)

    top = (y_max + y_min - length) / 2
    left = (x_max + x_min - length) / 2

    # top, left, width, height (return bbox as a square)
    return int(top), int(left), int(length), int(length)


def create_xml_point(root, name, x, y):
    part = root.createElement('part')
    part.setAttribute('name', name)
    part.setAttribute('x', str(x))
    part.setAttribute('y', str(y))

    return part


def create_xml_box(root, top, left, width, height, points):
    box = root.createElement('box')
    box.setAttribute('top', str(top))
    box.setAttribute('left', str(left))
    box.setAttribute('width', str(width))
    box.setAttribute('height', str(height))

    for i, point in enumerate(points):
        name = f"{i:02d}"
        box.appendChild(create_xml_point(root, name, *point))

    return box


def create_xml_image(root, filename, top, left, width, height, points):
    image = root.createElement('image')
    image.setAttribute('file', filename)

    box = create_xml_box(root, top, left, width, height, points)
    image.appendChild(box)

    return image


def create_xml(filenames: list, points: np.ndarray):
    root = minidom.Document()
    dataset = root.createElement('dataset')
    root.appendChild(dataset)

    images = root.createElement('images')
    dataset.appendChild(images)

    for filename, points_arr in zip(filenames, points):
        top, left, width, height = get_bbox(points_arr)

        image = create_xml_image(root, filename, top, left, width,
                                 height, points_arr)
        images.appendChild(image)

    with open(RESULT_XML, "w") as xml_file:
        root.writexml(xml_file)
