# CVAT facial landmarks annotation

68 facial keypoints annotation but can be modified to any number of keypoints

These project is intendes to facilitate the process of manual facial landmarks labeling with the help of CVAT tool.

1. Creating annotations for images
2. Creating annotations for videos

General pipeline:
- Process images to get .npy files with landmarks
- Generate .xml files with annotations that can be imported in CVAT
- Manually fix problematic keypoints
- Export from CVAT .xml with corrected annotations
- Extract new keypoints from xml  