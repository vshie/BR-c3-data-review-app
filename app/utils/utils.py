import json
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
from typing import List, Tuple, Dict, Any, Union


def rescale_instrinsic(
    k_mtx: np.ndarray, orig_shape: Tuple[int, int], img_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Rescales the intrinsic camera matrix based on new image dimensions.

    Args:
        k_mtx: The original intrinsic camera matrix (3x3 NumPy array).
        orig_shape: The original image shape (height, width).
        img_shape: The new image shape (height, width).

    Returns:
        The rescaled intrinsic camera matrix.
    """
    k_mtx[0, 0] = k_mtx[0, 0] * img_shape[1] / orig_shape[1]
    k_mtx[0, 2] = k_mtx[0, 2] * img_shape[1] / orig_shape[1]
    k_mtx[1, 1] = k_mtx[1, 1] * img_shape[0] / orig_shape[0]
    k_mtx[1, 2] = k_mtx[1, 2] * img_shape[0] / orig_shape[0]
    return k_mtx


def read_matrices(
    calib_data: Dict[str, Any],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tuple[int, int],
    Tuple[int, int],
    Tuple[int, int],
    np.ndarray,
    np.ndarray,
]:
    """
    Reads camera matrices and distortion coefficients from calibration data.

    Args:
        calib_data: A dictionary containing camera calibration data.

    Returns:
        A tuple containing intrinsic matrices, distortion coefficients, rectification transforms,
        original image shapes, and projection matrices for left, right, and center cameras.
    """
    for item in calib_data["cameraData"]:
        if item[0] == 1:
            k_mtx_left = np.array(item[1]["intrinsicMatrix"], dtype=np.float64)
            k_mtx_left = np.reshape(k_mtx_left, [3, 3])
            dist_left = np.array(item[1]["distortionCoeff"], dtype=np.float64)
            rot = np.array(item[1]["extrinsics"]["rotationMatrix"], dtype=np.float64)
            trans = np.array(
                [
                    item[1]["extrinsics"]["translation"]["x"],
                    item[1]["extrinsics"]["translation"]["y"],
                    item[1]["extrinsics"]["translation"]["z"],
                ],
                dtype=np.float64,
            )
            left_shape = (
                item[1]["height"],
                item[1]["width"],
            )
        elif item[0] == 2:
            k_mtx_right = np.array(item[1]["intrinsicMatrix"], dtype=np.float64)
            k_mtx_right = np.reshape(k_mtx_right, [3, 3])
            dist_right = np.array(item[1]["distortionCoeff"], dtype=np.float64)
            right_shape = [
                item[1]["height"],
                item[1]["width"],
            ]
        else:
            k_mtx_center = np.array(item[1]["intrinsicMatrix"], dtype=np.float64)
            k_mtx_center = np.reshape(k_mtx_center, [3, 3])
            dist_center = np.array(item[1]["distortionCoeff"], dtype=np.float64)
            center_shape = (
                item[1]["height"],
                item[1]["width"],
            )
            rect_center = np.array(np.eye(3), dtype=np.float64)

    rect_left, rect_right, p_left, p_right, Q, _, _ = cv2.stereoRectify(
        k_mtx_left,
        dist_left,
        k_mtx_right,
        dist_right,
        (left_shape[1], left_shape[0]),
        rot,
        (trans / 100).T,
        flags=cv2.CALIB_ZERO_DISPARITY,
    )

    return (
        k_mtx_left,
        k_mtx_right,
        k_mtx_center,
        dist_left,
        dist_right,
        dist_center,
        rect_left,
        rect_right,
        rect_center,
        left_shape,
        right_shape,
        center_shape,
        p_left,
        p_right,
    )


def undistortrectify(
    img: np.ndarray, calib_data: Dict[str, Any], instrument: str
) -> np.ndarray:
    """
    Undistorts and rectifies an image.

    Args:
        img: The input image (NumPy array).
        calib_data: Calibration data dictionary.
        instrument: Identifier for the camera ('left', 'right', or 'center').

    Returns:
        The undistorted and rectified image (NumPy array).
    """
    (
        k_mtx_l,
        k_mtx_r,
        k_mtx_c,
        dist_l,
        dist_r,
        dist_c,
        rect_l,
        rect_r,
        rect_c,
        shape_l,
        shape_r,
        shape_c,
        p_left,
        p_right,
    ) = read_matrices(calib_data)
    if len(img.shape) > 2:
        height, width, _ = img.shape
    else:
        height, width = img.shape

    if instrument == "left":
        k_mtx_l = rescale_instrinsic(k_mtx_l, shape_l, (height, width))
        map1, map2 = cv2.initUndistortRectifyMap(
            k_mtx_l, dist_l, rect_l, p_left, (width, height), cv2.CV_32FC1
        )
    elif instrument == "right":
        k_mtx_r = rescale_instrinsic(k_mtx_r, shape_r, (height, width))
        map1, map2 = cv2.initUndistortRectifyMap(
            k_mtx_r, dist_r, rect_r, p_right, (width, height), cv2.CV_32FC1
        )
    elif instrument == "center":
        k_mtx_c = rescale_instrinsic(k_mtx_c, shape_c, (height, width))
        map1, map2 = cv2.initUndistortRectifyMap(
            k_mtx_c, dist_c, rect_c, k_mtx_c[:3, :3], (width, height), m1type=0
        )
    image_rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    return image_rect


def triangulate_pts(
    calib_data: Dict[str, Any],
    left_pts: List[List[float]],
    right_pts: List[List[float]],
    img_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Triangulates 3D points from corresponding 2D points in rectified stereo images.
    Assumes points are already in the rectified image coordinate system.

    Args:
        calib_data: Calibration data dictionary.
        left_pts: A list of 2D points from the left rectified image. Each point is [x, y].
        right_pts: A list of 2D points from the right rectified image. Each point is [x, y].
        img_shape: The shape of the image (height, width). Currently unused.

    Returns:
        A NumPy array of triangulated 3D points.
    """
    (
        k_mtx_l,
        k_mtx_r,
        k_mtx_c,
        dist_l,
        dist_r,
        dist_c,
        rect_l,
        rect_r,
        rect_c,
        shape_l,
        shape_r,
        shape_c,
        p_left,
        p_right,
    ) = read_matrices(calib_data)
    left_ref_3d_pts = []
    if len(left_pts) != len(right_pts) or len(left_pts) % 2 != 0:
        raise ValueError("Number of left and right points must be equal and even.")

    for i, pt in enumerate(left_pts):
        left_ref_3d_pt = cv2.triangulatePoints(
            p_left, p_right, np.array(left_pts[i]).T, np.array(right_pts[i]).T
        )
        left_ref_3d_pt = (left_ref_3d_pt[:3] / left_ref_3d_pt[3]).T
        left_ref_3d_pts.append(left_ref_3d_pt)
    return np.array(left_ref_3d_pts)


def undistort_triangulate_pts(
    calib_data: Dict[str, Any],
    left_pts: List[List[float]],
    right_pts: List[List[float]],
    img_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Undistorts 2D points from original (distorted) stereo images and then triangulates them to 3D.

    Args:
        calib_data: Calibration data dictionary.
        left_pts: A list of 2D points from the original left image. Each point is [x, y].
        right_pts: A list of 2D points from the original right image. Each point is [x, y].
        img_shape: The shape of the image (height, width) from which the points were taken.

    Returns:
        A NumPy array of triangulated 3D points.
    """
    (
        k_mtx_l,
        k_mtx_r,
        k_mtx_c,
        dist_l,
        dist_r,
        dist_c,
        rect_l,
        rect_r,
        rect_c,
        shape_l,
        shape_r,
        shape_c,
        p_left,
        p_right,
    ) = read_matrices(calib_data)
    k_mtx_r = rescale_instrinsic(k_mtx_r, shape_r, img_shape)
    k_mtx_l = rescale_instrinsic(k_mtx_l, shape_l, img_shape)
    if len(left_pts) != len(right_pts) or len(left_pts) % 2 != 0:
        raise ValueError("Number of left and right points must be equal and even.")

    left_ref_3d_pts = []
    for i, pt in enumerate(left_pts):
        left_pt = cv2.undistortPoints(
            src=np.array(left_pts[i]),
            cameraMatrix=k_mtx_l,
            distCoeffs=dist_l,
            R=rect_l,
            P=p_left,
        )
        right_pt = cv2.undistortPoints(
            src=np.array(right_pts[i]),
            cameraMatrix=k_mtx_r,
            distCoeffs=dist_r,
            R=rect_r,
            P=p_right,
        )
        left_ref_3d_pt = cv2.triangulatePoints(
            p_left, p_right, np.array(left_pt), np.array(right_pt)
        )
        left_ref_3d_pt = (left_ref_3d_pt[:3] / left_ref_3d_pt[3]).T
        left_ref_3d_pts.append(left_ref_3d_pt)
    return left_ref_3d_pts


def undistort_pts(
    calib_data: Dict[str, Any],
    pts: List[List[float]],
    instrument: str,
    img_shape: Tuple[int, int],
) -> List[np.ndarray]:
    """
    Undistorts 2D points from an original (distorted) image to their rectified positions.

    Args:
        calib_data: Calibration data dictionary.
        pts: A list of 2D points from the original image. Each point is [x, y].
        instrument: Identifier for the camera ('Oak1Left' or 'Oak1Right').
        img_shape: The shape of the image (height, width) from which the points were taken.
    Returns:
        A list of NumPy arrays, where each array contains the undistorted [x, y] coordinates.
    """
    (
        k_mtx_l,
        k_mtx_r,
        k_mtx_c,
        dist_l,
        dist_r,
        dist_c,
        rect_l,
        rect_r,
        rect_c,
        shape_l,
        shape_r,
        shape_c,
        p_left,
        p_right,
    ) = read_matrices(calib_data)
    rect_pts = []
    for i, pt in enumerate(pts):
        if instrument == "Oak1Left":
            k_mtx_l = rescale_instrinsic(k_mtx_l, shape_l, img_shape)
            rect_pt = cv2.undistortPoints(
                src=np.array(pt),
                cameraMatrix=k_mtx_l,
                distCoeffs=dist_l,
                R=rect_l,
                P=p_left,
            )
            rect_pts.append(rect_pt.flatten())
        elif instrument == "Oak1Right":
            k_mtx_r = rescale_instrinsic(k_mtx_r, shape_r, img_shape)
            rect_pt = cv2.undistortPoints(
                src=np.array(pt),
                cameraMatrix=k_mtx_r,
                distCoeffs=dist_r,
                R=rect_r,
                P=p_right,
            )
            rect_pts.append(rect_pt.flatten())
    return rect_pts


def redistort_pts(
    calib_data: Dict[str, Any],
    pts: List[List[float]],
    instrument: str,
    img_shape: Tuple[int, int],
) -> List[np.ndarray]:
    """
    Projects 2D points from a rectified image back to their original (distorted) image positions.

    Args:
        calib_data: Calibration data dictionary.
        pts: A list of 2D points from the rectified image. Each point is [x, y].
        instrument: Identifier for the camera ('Oak1Left' or 'Oak1Right').
        img_shape: The shape of the image (height, width) to which the points are being re-projected.
    Returns:
        A list of NumPy arrays, where each array contains the re-distorted [x, y] coordinates.
    """
    (
        k_mtx_l,
        k_mtx_r,
        k_mtx_c,
        dist_l,
        dist_r,
        dist_c,
        rect_l,
        rect_r,
        rect_c,
        shape_l,
        shape_r,
        shape_c,
        p_left,
        p_right,
    ) = read_matrices(calib_data)
    un_rect_pts = []
    for i, pt in enumerate(pts):
        if instrument == "Oak1Left":
            k_mtx_l = rescale_instrinsic(k_mtx_l, shape_l, img_shape)
            p_left = rescale_instrinsic(p_left, shape_l, img_shape)
            rect_pt = cv2.undistortPoints(
                src=np.array(pt),
                cameraMatrix=p_left[:3, :3],
                distCoeffs=np.zeros(5),
            )
            un_rect_pt, _ = cv2.projectPoints(
                objectPoints=cv2.convertPointsToHomogeneous(rect_pt),
                rvec=np.eye(3),
                tvec=np.zeros(3),
                cameraMatrix=k_mtx_l,
                distCoeffs=dist_l,
            )
            un_rect_pts.append(un_rect_pt.flatten())
        elif instrument == "Oak1Right":
            k_mtx_r = rescale_instrinsic(k_mtx_r, shape_r, img_shape)
            p_right = rescale_instrinsic(p_right, shape_r, img_shape)
            rect_pt = cv2.undistortPoints(
                src=np.array(pt),
                cameraMatrix=p_right[:3, :3],
                distCoeffs=np.zeros(5),
            )
            un_rect_pt, _ = cv2.projectPoints(
                objectPoints=cv2.convertPointsToHomogeneous(rect_pt),
                rvec=np.eye(3),
                tvec=np.zeros(3),
                cameraMatrix=k_mtx_r,
                distCoeffs=dist_r,
            )
            un_rect_pts.append(un_rect_pt.flatten())
    return un_rect_pts


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculates the Euclidean distance between two 3D points.

    Args:
        point1: A NumPy array representing the first 3D point [x, y, z].
        point2: A NumPy array representing the second 3D point [x, y, z].

    Returns:
        The Euclidean distance between the two points.
    """
    return np.linalg.norm(np.subtract(point1, point2))


def calculate_size(calib_pts: List[np.ndarray]) -> List[float]:
    """
    Calculates distances between pairs of 3D points.

    Args:
        calib_pts: A list of 3D points (NumPy arrays). It's assumed that points
                   are paired (e.g., [pt1_start, pt1_end, pt2_start, pt2_end, ...]).

    Returns:
        A list of calculated distances, rounded to 3 decimal places.
    """
    distances = []
    for i in range(0, len(calib_pts), 2):
        distance = calculate_distance(calib_pts[i], calib_pts[i + 1])
        distances.append(round(distance, 3))
    return distances


def add_annotations(
    annotations: Union[pd.DataFrame, List[Dict[str, Any]], None],
    fig: go.Figure,
    instrument: str,
    name: str,
    rectify: bool,
    calib_data: Dict[str, Any],
    img_shape: Tuple[int, int],
) -> go.Figure:
    """
    Adds annotations (lines or rectangles with labels) to a Plotly figure based on stored annotation data.
    Handles transformation of annotation coordinates if the rectification state changes.

    Args:
        annotations: A Pandas DataFrame or list of dictionaries containing annotation data, or None.
        fig: The Plotly figure object to add annotations to.
        instrument: The identifier of the camera/instrument for which to filter annotations.
        name: The name/identifier of the specific image file to filter annotations.
        rectify: A boolean indicating whether the current view is rectified.
        calib_data: Calibration data dictionary, used for transforming points if rectification state differs.
        img_shape: The shape (height, width) of the image in the figure.

    Returns:
        The Plotly figure object with added annotations.
    """
    if annotations is not None and annotations != []:
        annotations = pd.DataFrame(annotations)
        rows = annotations[
            (annotations["instrument"] == instrument) & (annotations["name"] == name)
        ]
        if not rows.empty:
            for index, row in rows.iterrows():
                values = row["values"]
                locations = row["location"]
                if row["type"] == "distance":
                    if values and locations:
                        if row["rectified"] != rectify:
                            if row["rectified"] == True:
                                # # This works OK but not well enough to include for now.
                                # locations = redistort_pts(
                                #     calib_data, locations, instrument, img_shape
                                # )
                                continue
                            else:
                                locations = undistort_pts(
                                    calib_data, locations, instrument, img_shape
                                )
                        shape_exists = False
                        if fig.layout.shapes:
                            for shape in fig.layout.shapes:
                                if (
                                    shape.type == "line"
                                    and shape.x0 == locations[0][0]
                                    and shape.y0 == locations[0][1]
                                    and shape.x1 == locations[1][0]
                                    and shape.y1 == locations[1][1]
                                    and shape.label.text == str(values) + " m"
                                ):
                                    shape_exists = True
                                    break
                        if not shape_exists:
                            fig.add_shape(
                                type="line",
                                x0=locations[0][0],
                                y0=locations[0][1],
                                x1=locations[1][0],
                                y1=locations[1][1],
                                line_width=3,
                                line_color="aquamarine",
                                label=dict(
                                    text=str(values) + " m",
                                    font=dict(color="aquamarine"),
                                ),
                            )

                elif row["type"] == "comment":
                    if values and locations:
                        if row["rectified"] != rectify:
                            if row["rectified"] == True:
                                continue
                            else:
                                locations = undistort_pts(
                                    calib_data, locations, instrument, img_shape
                                )
                        shape_exists = False
                        if fig.layout.shapes:
                            for shape in fig.layout.shapes:
                                if (
                                    shape.type == "rect"
                                    and shape.x0 == locations[0][0]
                                    and shape.y0 == locations[0][1]
                                    and shape.x1 == locations[1][0]
                                    and shape.y1 == locations[1][1]
                                    and shape.label.text == str(values)
                                ):
                                    shape_exists = True
                                    break
                        if not shape_exists:
                            fig.add_shape(
                                type="rect",
                                x0=locations[0][0],
                                y0=locations[0][1],
                                x1=locations[1][0],
                                y1=locations[1][1],
                                line=dict(width=2, color="aquamarine"),
                                label=dict(
                                    text=str(values),
                                    font=dict(color="aquamarine"),
                                    textposition="top left",
                                ),
                            )
    return fig
