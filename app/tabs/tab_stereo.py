from dash import (
    Dash,
    html,
    dcc,
    callback,
    Input,
    Output,
    State,
    no_update,
    ctx,
    get_asset_url,
)
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import dash_extensions as de
import plotly.graph_objects as go
from utils.utils import (
    undistortrectify,
    triangulate_pts,
    calculate_size,
    undistort_triangulate_pts,
    add_annotations,
    configure_plots,
)
from typing import List, Tuple, Dict, Any, Optional, Union


class StereoOakTab:
    """
    Represents the 'Stereo Cameras' tab in the Dash application.
    This tab displays images from the left and right stereo cameras.
    """

    def __init__(self, tab_suffix: str, config: Dict[str, Any]):
        """
        Initializes the StereoOakTab.

        Args:
            tab_suffix: A string suffix to make component IDs unique (e.g., "stereo").
            config: Configuration dictionary for Plotly graphs.
        """
        self.suffix = tab_suffix
        self.config = config

    def create_layout(self) -> html.Div:
        """
        Creates the layout for the 'Stereo Cameras' tab.
        Returns:
            A Dash HTML Div component containing the tab's layout.
        """
        layout = html.Div(
            [
                dbc.Row(style={"height": "1vh"}),
                dbc.Row(
                    [
                        dbc.Modal(
                            [
                                dbc.ModalHeader(dbc.ModalTitle("Add Note: ")),
                                dbc.ModalBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Textarea(
                                                        id="note_{}".format(
                                                            self.suffix
                                                        ),
                                                    ),
                                                    width=9,
                                                ),
                                                dbc.Col(
                                                    dbc.Button(
                                                        "Submit",
                                                        id="add_note_{}".format(
                                                            self.suffix
                                                        ),
                                                        n_clicks=0,
                                                        class_name="button",
                                                    ),
                                                    width=2,
                                                    align="end",
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                            ],
                            id="note_modal_{}".format(self.suffix),
                            is_open=False,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Load Data",
                                id="load_{}".format(self.suffix),
                                n_clicks=0,
                                class_name="button",
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Button(
                                [
                                    html.I(
                                        className="bi bi-rulers",
                                        style={"color": "#f0a120"},
                                    ),
                                    " Measure",
                                ],
                                id="size_{}".format(self.suffix),
                                # n_clicks=0,
                                class_name="button",
                            ),
                            width="auto",
                        ),
                        dcc.Store(id="index_{}".format(self.suffix)),
                        dcc.Store(id="alert_state_{}".format(self.suffix)),
                    ],
                    justify="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                id="OakLeft_id_{}".format(self.suffix),
                                style={"font-size": "10px"},
                            )
                        ),
                        dcc.Store(id="left_shape_{}".format(self.suffix)),
                        dcc.Store(id="right_shape_{}".format(self.suffix)),
                        dcc.Store(id="shape_names_{}".format(self.suffix)),
                        dbc.Col(
                            html.Div(
                                id="OakRight_id_{}".format(self.suffix),
                                style={"font-size": "10px"},
                            )
                        ),
                    ],
                    class_name="g-0",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="OakLeft_{}".format(self.suffix),
                                responsive=True,
                                clear_on_unhover=True,
                                config=self.config,
                                style={
                                    "width": "100%",
                                    "height": "70vh",
                                },
                            )
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id="OakRight_{}".format(self.suffix),
                                responsive=True,
                                config=self.config,
                                clear_on_unhover=True,
                                style={
                                    "width": "100%",
                                    "height": "70vh",
                                },
                            )
                        ),
                    ],
                    class_name="g-0",
                    justify="center",
                ),
                dbc.Row(style={"height": "1vh"}),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button(
                                            children=[
                                                html.I(
                                                    className="bi bi-caret-left-fill",
                                                    style={
                                                        "color": "#f0a120",
                                                    },
                                                ),
                                                " Previous",
                                            ],
                                            id="previous_{}".format(self.suffix),
                                            style={"width": "10vw"},
                                            className="button",
                                        ),
                                        dbc.Button(
                                            children=[
                                                "Next ",
                                                html.I(
                                                    className="bi bi-caret-right-fill",
                                                    style={"color": "#f0a120"},
                                                ),
                                            ],
                                            id="next_{}".format(self.suffix),
                                            style={"width": "10vw"},
                                            className="button",
                                        ),
                                    ],
                                )
                            ],
                            width="auto",
                        ),
                    ],
                    justify="center",
                ),
                dbc.Row(
                    [
                        de.Keyboard(
                            captureKeys=["ArrowLeft"],
                            id="previous_key_{}".format(self.suffix),
                        ),
                        de.Keyboard(
                            captureKeys=["ArrowRight"],
                            id="next_key_{}".format(self.suffix),
                        ),
                    ]
                ),
            ],
            style={"background-color": "white"},
        )
        return layout


class StereoOakCallbacks:
    """
    Manages all the callbacks for the 'Stereo Cameras' tab.
    """

    def __init__(self, tab_suffix: str, app: Dash):
        """
        Initializes the StereoOakCallbacks.

        Args:
            tab_suffix: The string suffix used for component IDs in this tab.
            app: The main Dash application instance.
        """
        self.suffix = tab_suffix
        self.app = app

    def create_callbacks(self) -> None:
        """
        Defines and registers all callbacks for the 'Stereo Cameras' tab.
        """

        @self.app.callback(
            Output("OakLeft_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("OakRight_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output(
                "OakLeft_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output(
                "OakRight_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output("left_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            Output("right_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            Output("alert_state_{}".format(self.suffix), "data", allow_duplicate=True),
            State("data_paths", "data"),
            State("calibration", "data"),
            State("annotations", "data"),
            State("rectify", "value"),
            Input("load_{}".format(self.suffix), "n_clicks"),
        )
        def display_stereo(data, calibration_data, annotations, rectify, n_clicks):
            """
            Loads and displays the initial set of stereo images (left and right)
            when the 'Load Data' button is clicked.
            Handles image rectification if enabled and calibration data is available.
            Adds existing annotations to the figures.
            """
            if n_clicks is None or data is None:
                raise PreventUpdate
            else:
                path_data = pd.DataFrame(data)
                img1 = Image.open(path_data["Oak1Left"].iloc[0])
                r_times = pd.to_datetime(path_data["Oak1RightTimes"])
                l_times = pd.to_datetime(path_data["Oak1LeftTimes"])
                if abs((r_times.iloc[0] - l_times.iloc[0]).total_seconds()) < 0.05:
                    img2 = Image.open(path_data["Oak1Right"].iloc[0])
                    alert_state = no_update
                else:
                    img2 = Image.open("assets/MS_Full_White.png")
                    alert_state = {
                        "state": True,
                        "msg": "Could not find right image with timestamp within 5 ms of left image.",
                    }
                if rectify:
                    if calibration_data is not None:
                        img1 = undistortrectify(
                            np.array(img1), calibration_data, "left"
                        )
                        img2 = undistortrectify(
                            np.array(img2), calibration_data, "right"
                        )
                fig1 = px.imshow(
                    img1, color_continuous_scale="gray", binary_string=True
                )
                fig1 = configure_plots(fig1, rectify)

                fig2 = px.imshow(
                    img2, color_continuous_scale="gray", binary_string=True
                )
                fig2 = configure_plots(fig2, rectify)

                fig1 = add_annotations(
                    annotations,
                    fig1,
                    "Oak1Left",
                    path_data["Oak1Left"].iloc[0],
                    rectify,
                    calibration_data,
                    np.array(img1).shape,
                )
                fig2 = add_annotations(
                    annotations,
                    fig2,
                    "Oak1Right",
                    path_data["Oak1Right"].iloc[0],
                    rectify,
                    calibration_data,
                    np.array(img2).shape,
                )
                return (
                    fig1,
                    fig2,
                    path_data["Oak1Left"].iloc[0],
                    path_data["Oak1Right"].iloc[0],
                    np.array(img1).shape,
                    np.array(img2).shape,
                    alert_state,
                )

        @self.app.callback(
            Output("OakLeft_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("OakRight_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output(
                "OakLeft_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output(
                "OakRight_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output("left_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            Output("right_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            Output("alert_state_{}".format(self.suffix), "data", allow_duplicate=True),
            State("data_paths", "data"),
            State("calibration", "data"),
            State("annotations", "data"),
            State("rectify", "value"),
            Input("index_{}".format(self.suffix), "data"),
        )
        def update_stereo(data, calibration_data, annotations, rectify, index):
            """
            Updates the displayed stereo images (left and right) when the image index changes
            (e.g., via Next/Previous buttons).
            Handles image rectification and adds annotations.
            """
            if index is None or data is None:
                raise PreventUpdate
            path_data = pd.DataFrame(data)
            img1 = Image.open(path_data["Oak1Left"].iloc[index])
            r_times = pd.to_datetime(path_data["Oak1RightTimes"])
            l_times = pd.to_datetime(path_data["Oak1LeftTimes"])
            idx = np.argmin(abs(l_times.iloc[index] - r_times))
            if abs((l_times.iloc[index] - r_times.iloc[idx]).total_seconds()) > 0.05:
                img2 = Image.open("assets/MS_Full_White.png")
                alert_state = {
                    "state": True,
                    "msg": "Could not find right image with timestamp within 5 ms of left image.",
                }
            else:
                img2 = Image.open(path_data["Oak1Right"].iloc[idx])
                alert_state = no_update
            if rectify:
                if calibration_data is not None:
                    img1 = undistortrectify(np.array(img1), calibration_data, "left")
                    img2 = undistortrectify(np.array(img2), calibration_data, "right")
            fig1 = px.imshow(img1, color_continuous_scale="gray", binary_string=True)
            fig1 = configure_plots(fig1, rectify)
            fig2 = px.imshow(img2, color_continuous_scale="gray", binary_string=True)
            fig2 = configure_plots(fig2, rectify)
            fig1 = add_annotations(
                annotations,
                fig1,
                "Oak1Left",
                path_data["Oak1Left"].iloc[index],
                rectify,
                calibration_data,
                np.array(img1).shape,
            )
            fig2 = add_annotations(
                annotations,
                fig2,
                "Oak1Right",
                path_data["Oak1Right"].iloc[idx],
                rectify,
                calibration_data,
                np.array(img2).shape,
            )
            return (
                fig1,
                fig2,
                path_data["Oak1Left"].iloc[index],
                path_data["Oak1Right"].iloc[idx],
                np.array(img1).shape,
                np.array(img2).shape,
                alert_state,
            )

        @self.app.callback(
            Output("OakLeft_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("OakRight_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output(
                "OakLeft_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output(
                "OakRight_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output("alert_state_{}".format(self.suffix), "data", allow_duplicate=True),
            Output("left_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            Output("right_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            State("data_paths", "data"),
            State("calibration", "data"),
            State("annotations", "data"),
            State("index_{}".format(self.suffix), "data"),
            Input("rectify", "value"),
            prevent_initial_call=True,
        )
        def rectify_stereo(data, calibration_data, annotations, index, rectify):
            """
            Updates the displayed stereo images (left and right) when the 'Rectify' switch changes.
            Applies or removes rectification and updates annotations accordingly.
            Shows an alert if rectification is attempted without calibration data.
            """
            if data is None:
                raise PreventUpdate
            if index is None:
                index = 0
            path_data = pd.DataFrame(data)
            img1 = Image.open(path_data["Oak1Left"].iloc[index])
            r_times = pd.to_datetime(path_data["Oak1RightTimes"])
            l_times = pd.to_datetime(path_data["Oak1LeftTimes"])
            idx = np.argmin(abs(l_times.iloc[index] - r_times))
            if abs((l_times.iloc[index] - r_times.iloc[idx]).total_seconds()) > 0.05:
                img2 = Image.open("assets/MS_Full_White.png")
                alert_state = {
                    "state": True,
                    "msg": "Could not find right image with timestamp within 5 ms of left image.",
                }
            else:
                img2 = Image.open(path_data["Oak1Right"].iloc[idx])
                alert_state = no_update
            if rectify:
                if calibration_data is not None:
                    img1 = undistortrectify(np.array(img1), calibration_data, "left")
                    img2 = undistortrectify(np.array(img2), calibration_data, "right")
            fig1 = px.imshow(img1, color_continuous_scale="gray", binary_string=True)
            fig1 = configure_plots(fig1, rectify)
            fig2 = px.imshow(img2, color_continuous_scale="gray", binary_string=True)
            fig2 = configure_plots(fig2, rectify)
            fig1 = add_annotations(
                annotations,
                fig1,
                "Oak1Left",
                path_data["Oak1Left"].iloc[index],
                rectify,
                calibration_data,
                np.array(img1).shape,
            )
            fig2 = add_annotations(
                annotations,
                fig2,
                "Oak1Right",
                path_data["Oak1Right"].iloc[idx],
                rectify,
                calibration_data,
                np.array(img2).shape,
            )
            if calibration_data is None:
                return (
                    fig1,
                    fig2,
                    path_data["Oak1Left"].iloc[index],
                    path_data["Oak1Right"].iloc[idx],
                    {
                        "state": True,
                        "msg": "Looks like you're missing a calibration file. Upload one using the selector above.",
                    },
                    np.array(img1).shape,
                    np.array(img2).shape,
                )
            else:
                return (
                    fig1,
                    fig2,
                    path_data["Oak1Left"].iloc[index],
                    path_data["Oak1Right"].iloc[idx],
                    alert_state,
                    np.array(img1).shape,
                    np.array(img2).shape,
                )

        @self.app.callback(
            Output("OakLeft_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("OakRight_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("annotations", "data", allow_duplicate=True),
            Output("alert_state_{}".format(self.suffix), "data", allow_duplicate=True),
            State("calibration", "data"),
            State("rectify", "value"),
            State("annotations", "data"),
            State("OakLeft_{}".format(self.suffix), "figure"),
            State("OakRight_{}".format(self.suffix), "figure"),
            State("OakLeft_id_{}".format(self.suffix), "children"),
            State("OakRight_id_{}".format(self.suffix), "children"),
            # State("OakLeft_{}".format(self.suffix), "relayoutData"),
            # State("OakRight_{}".format(self.suffix), "relayoutData"),
            # State("OakLeft_{}".format(self.suffix), "clickData"),
            State("left_shape_{}".format(self.suffix), "data"),
            State("right_shape_{}".format(self.suffix), "data"),
            State("shape_names_{}".format(self.suffix), "data"),
            Input("size_{}".format(self.suffix), "n_clicks"),
            prevent_initial_call=True,
        )
        def get_size(
            calibration_data,
            rectify,
            annotations,
            left_fig,
            right_fig,
            left_name,
            right_name,
            # clickData,
            # leftData,
            # rightData,
            left_shape,
            right_shape,
            shape_names,
            n_clicks,
        ):
            """
            Calculates distances between points drawn on the left and right stereo images
            and displays these distances on the images.
            Requires calibration data. Shows an alert if missing.
            Handles both rectified and unrectified points.

            Args:
                calibration_data: Loaded camera calibration data.
                rectify: Boolean indicating if images are currently rectified.
                annotations: Current list of annotations.
                left_fig, right_fig: Current Plotly figures for left and right images.
                left_name, right_name: File names of the current left and right images.
                leftData, rightData: relayoutData from left and right figures, containing drawn shapes.
                left_shape, right_shape: Shapes of the current left and right images.
                n_clicks: Number of times the 'Measure' button was clicked.
            """
            if n_clicks is None:
                raise PreventUpdate
            if calibration_data is None:
                # raise PreventUpdate
                return (
                    no_update,
                    no_update,
                    no_update,
                    {
                        "state": True,
                        "msg": "Looks like you're missing a calibration file. Upload one using the selector above.",
                    },
                )
            leftData = left_fig["layout"]["shapes"]
            rightData = right_fig["layout"]["shapes"]
            if not leftData and not rightData:
                raise PreventUpdate

            # if (
            #     "shapes" not in leftData
            #     or "shapes" not in rightData
            #     or not leftData["shapes"]
            #     or not rightData["shapes"]
            #     # or not any(["shapes" in key for key in leftData])
            #     # or not any(["shapes" in key for key in rightData])
            # ):
            #     raise PreventUpdate
            left_pts = []
            right_pts = []
            left_count = 0
            right_count = 0

            left_fig = go.Figure(left_fig)
            right_fig = go.Figure(right_fig)

            if annotations is None:
                annotations = []

            for item in leftData:
                if (
                    item["type"] == "line"
                    and "editable" in item
                    and item["editable"] is True
                ):
                    left_exists = any(
                        ann["name"] == left_name
                        and ann["instrument"] == "Oak1Left"
                        and ann["location"]
                        == [[item["x0"], item["y0"]], [item["x1"], item["y1"]]]
                        for ann in annotations
                    )
                    if not left_exists:
                        left_pts.append([item["x0"], item["y0"]])
                        left_pts.append([item["x1"], item["y1"]])
                        left_count += 1
            for item in rightData:
                if (
                    item["type"] == "line"
                    and "editable" in item
                    and item["editable"] is True
                ):
                    right_exists = any(
                        ann["name"] == right_name
                        and ann["instrument"] == "Oak1Right"
                        and ann["location"]
                        == [[item["x0"], item["y0"]], [item["x1"], item["y1"]]]
                        for ann in annotations
                    )
                    if not right_exists:
                        right_pts.append([item["x0"], item["y0"]])
                        right_pts.append([item["x1"], item["y1"]])
                        right_count += 1

            if left_count != right_count:
                return (
                    no_update,
                    no_update,
                    no_update,
                    {
                        "state": True,
                        "msg": "Unequal number of measurement lines on left and right images. Can't calculate size.",
                    },
                )

            if rectify:
                pts = triangulate_pts(calibration_data, left_pts, right_pts, left_shape)
            else:
                pts = undistort_triangulate_pts(
                    calibration_data, left_pts, right_pts, left_shape
                )
            distances = calculate_size(pts)

            # Clear existing shapes before adding new ones
            if left_fig.layout.shapes:
                left_fig.layout.shapes = [
                    s
                    for s in left_fig.layout.shapes
                    if (
                        s.name == "measured_shape_left"
                        or s.name == "saved_shape"
                        or s.name == "annotated_rect_left"
                    )
                ]
            else:
                left_fig.layout.shapes = []
            if right_fig.layout.shapes:
                right_fig.layout.shapes = [
                    s
                    for s in right_fig.layout.shapes
                    if (
                        s.name == "measured_shape_right"
                        or s.name == "saved_shape"
                        or s.name == "annotated_rect_right"
                    )
                ]
            else:
                right_fig.layout.shapes = []
            left_fig.data = [
                trace
                for trace in left_fig.data
                if not (
                    hasattr(trace, "meta")
                    and trace.meta
                    and trace.meta.get("type") == "line_endpoint_marker_left"
                )
            ]
            right_fig.data = [
                trace
                for trace in right_fig.data
                if not (
                    hasattr(trace, "meta")
                    and trace.meta
                    and trace.meta.get("type") == "line_endpoint_marker_right"
                )
            ]
            if left_fig.layout.annotations:
                left_fig.layout.annotations = [
                    s
                    for s in left_fig.layout.annotations
                    if not (s.name == "line_endpoint_text_left")
                ]
            else:
                left_fig.layout.annotations = []

            for i, distance in enumerate(distances):
                loc = left_pts[i * 2 : i * 2 + 2]
                text_ann = [
                    shape_names[ann_key]["name"]
                    for ann_key in list(shape_names.keys())
                    if shape_names[ann_key]["loc"]
                    == [(loc[0][0] + loc[1][0]) / 2, (loc[0][1] + loc[1][1]) / 2]
                ]
                if not text_ann:
                    text_ann = ["<edit name>"]
                if text_ann[0] == "<edit name>":
                    text_ann = [""]
                else:
                    text_ann[0] = text_ann[0] + ": "
                left_ann = {
                    "name": left_name,
                    "instrument": "Oak1Left",
                    "type": "distance",
                    "rectified": rectify,
                    "location": left_pts[i * 2 : i * 2 + 2],
                    "values": text_ann[0] + str(distances[i]) + " m",
                }

                right_ann = {
                    "name": right_name,
                    "instrument": "Oak1Right",
                    "type": "distance",
                    "rectified": rectify,
                    "location": right_pts[i * 2 : i * 2 + 2],
                    "values": text_ann[0] + str(distances[i]) + " m",
                }
                left_exists = any(
                    ann["name"] == left_name
                    and ann["instrument"] == "Oak1Left"
                    and ann["location"] == left_pts[i * 2 : i * 2 + 2]
                    for ann in annotations
                )
                if not left_exists:
                    annotations.append(left_ann)
                    left_fig.add_shape(
                        type="line",
                        editable=False,
                        x0=left_pts[i * 2][0],
                        y0=left_pts[i * 2][1],
                        x1=left_pts[i * 2 + 1][0],
                        y1=left_pts[i * 2 + 1][1],
                        line_width=3,
                        line_color="red",
                        label=dict(
                            text=text_ann[0] + str(distances[i]) + " m",
                            font=dict(color="red"),
                        ),
                        name="measured_shape_left",
                    )
                right_exists = any(
                    ann["name"] == right_name
                    and ann["instrument"] == "Oak1Right"
                    and ann["location"] == right_pts[i * 2 : i * 2 + 2]
                    for ann in annotations
                )
                if not right_exists:
                    annotations.append(right_ann)
                    right_fig.add_shape(
                        type="line",
                        editable=False,
                        x0=right_pts[i * 2][0],
                        y0=right_pts[i * 2][1],
                        x1=right_pts[i * 2 + 1][0],
                        y1=right_pts[i * 2 + 1][1],
                        line_width=3,
                        line_color="red",
                        label=dict(
                            text=text_ann[0] + str(distances[i]) + " m",
                            font=dict(color="red"),
                        ),
                        name="measured_shape_right",
                    )
            return left_fig, right_fig, annotations, no_update

        @self.app.callback(
            Output(
                "note_modal_{}".format(self.suffix), "is_open", allow_duplicate=True
            ),
            Output("OakLeft_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("OakRight_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("shape_names_{}".format(self.suffix), "data", allow_duplicate=True),
            State("annotations", "data"),
            State("OakLeft_id_{}".format(self.suffix), "children"),
            State("OakRight_id_{}".format(self.suffix), "children"),
            State("OakLeft_{}".format(self.suffix), "figure"),
            State("OakRight_{}".format(self.suffix), "figure"),
            State("rectify", "value"),
            State("shape_names_{}".format(self.suffix), "data"),
            Input("OakLeft_{}".format(self.suffix), "relayoutData"),
            Input("OakRight_{}".format(self.suffix), "relayoutData"),
            prevent_initial_call=True,
        )
        def capture_relayout_events(
            annotations,
            left_name,
            right_name,
            left_fig,
            right_fig,
            rectify,
            shape_names,
            leftData,
            rightData,
        ):
            """
            Opens the note modal if a new rectangular shape (for commenting) is drawn
            on either the left or right stereo image and that shape
            has not already been annotated.

            Args:
                annotations: Current list of annotations.
                left_name, right_name: File names of the current left and right images.
                leftData, rightData: relayoutData from the left and right image figures.
            """
            left_fig = go.Figure(left_fig)
            right_fig = go.Figure(right_fig)
            if leftData is None and rightData is None:
                raise PreventUpdate

            modal_should_open = False
            figure_was_updated = False

            if annotations is None:
                annotations = []
            if shape_names is None:
                shape_names = {}

            if leftData and any(k.startswith("annotations[") for k in leftData):
                k = [k for k in leftData if k.startswith("annotations[")][0]
                index = int(k.replace("annotations[", " ").replace("].text", " "))
                left_fig.update_annotations(dict(text=leftData[k]), selector=index)
                # shape_names.append({"name": leftData[k], "index": index})
                shape_names[str(index)]["name"] = leftData[k]
                return (no_update, left_fig, right_fig, shape_names)

            if leftData and (
                "shapes" in leftData or any(k.startswith("shapes[") for k in leftData)
            ):
                figure_was_updated = True
                left_fig.data = [
                    trace
                    for trace in left_fig.data
                    if not (
                        hasattr(trace, "meta")
                        and trace.meta
                        and trace.meta.get("type") == "line_endpoint_marker_left"
                    )
                ]
                if right_fig.layout.shapes:
                    right_fig.layout.shapes = [
                        s
                        for s in right_fig.layout.shapes
                        if not (s.name == "left_epipolar_line")
                    ]
                else:
                    right_fig.layout.shapes = []
                if left_fig.layout.annotations:
                    left_fig.layout.annotations = [
                        s
                        for s in left_fig.layout.annotations
                        if not (s.name == "line_endpoint_text_left")
                    ]
                else:
                    left_fig.layout.annotations = []
                # for item in leftData["shapes"]:
                ann_idx = 0
                if left_fig.layout.shapes:
                    for item in left_fig.layout.shapes:
                        if item.type == "line" and item.editable is True:
                            left_fig.add_trace(
                                go.Scatter(
                                    x=[item["x0"], item["x1"]],
                                    y=[item["y0"], item["y1"]],
                                    mode="markers",
                                    marker=dict(color="cyan", size=8),
                                    showlegend=False,
                                    meta={"type": "line_endpoint_marker_left"},
                                )
                            )
                            text_ann = [
                                shape_names[ann_key]["name"]
                                for ann_key in list(shape_names.keys())
                                if shape_names[ann_key]["loc"]
                                == [
                                    (item["x0"] + item["x1"]) / 2,
                                    (item["y0"] + item["y1"]) / 2,
                                ]
                            ]
                            if not text_ann:
                                text_ann = ["<edit name>"]
                            left_fig.add_annotation(
                                x=(item["x0"] + item["x1"]) / 2,
                                y=(item["y0"] + item["y1"]) / 2,
                                showarrow=False,
                                text=text_ann[0],
                                font=dict(color="cyan"),
                                name="line_endpoint_text_left",
                            )
                            shape_names[str(ann_idx)] = {
                                "name": text_ann[0],
                                "loc": [
                                    (item["x0"] + item["x1"]) / 2,
                                    (item["y0"] + item["y1"]) / 2,
                                ],
                            }
                            ann_idx += 1
                            if rectify:
                                right_fig.add_hline(
                                    y=item["y0"],
                                    line=dict(color="blue", dash="dash", width=1),
                                    editable=False,
                                    name="left_epipolar_line",
                                )
                                right_fig.add_hline(
                                    y=item["y1"],
                                    line=dict(color="blue", dash="dash", width=1),
                                    editable=False,
                                    name="left_epipolar_line",
                                )
                if "shapes" in leftData and isinstance(leftData["shapes"], list):
                    for item_new in leftData["shapes"]:
                        if item_new["type"] == "rect":

                            # elif item["type"] == "rect":
                            left_exists = any(
                                ann["name"] == left_name
                                and ann["instrument"] == "Oak1Left"
                                and ann["location"]
                                == [
                                    [item_new["x0"], item_new["y0"]],
                                    [item_new["x1"], item_new["y1"]],
                                ]
                                for ann in annotations
                            )
                            if not left_exists:
                                # return True, left_fig, right_fig
                                modal_should_open = True
                                break
            if rightData and (
                "shapes" in rightData or any(k.startswith("shapes[") for k in rightData)
            ):
                figure_was_updated = True
                right_fig.data = [
                    trace
                    for trace in right_fig.data
                    if not (
                        hasattr(trace, "meta")
                        and trace.meta
                        and trace.meta.get("type") == "line_endpoint_marker_right"
                    )
                ]
                if left_fig.layout.shapes:
                    left_fig.layout.shapes = [
                        s
                        for s in left_fig.layout.shapes
                        if not (s.name == "right_epipolar_line")
                    ]
                else:
                    left_fig.layout.shapes = []
                if right_fig.layout.shapes:
                    for item in right_fig.layout.shapes:
                        if item.type == "line" and item.editable is True:
                            right_fig.add_trace(
                                go.Scatter(
                                    x=[item["x0"], item["x1"]],
                                    y=[item["y0"], item["y1"]],
                                    mode="markers",
                                    marker=dict(color="cyan", size=8),
                                    showlegend=False,
                                    meta={"type": "line_endpoint_marker_right"},
                                )
                            )
                            if rectify:
                                left_fig.add_hline(
                                    y=item["y0"],
                                    line=dict(color="blue", dash="dash", width=1),
                                    editable=False,
                                    name="right_epipolar_line",
                                )
                                left_fig.add_hline(
                                    y=item["y1"],
                                    line=dict(color="blue", dash="dash", width=1),
                                    editable=False,
                                    name="right_epipolar_line",
                                )
                if (
                    not modal_should_open
                    and "shapes" in rightData
                    and isinstance(rightData["shapes"], list)
                ):
                    for item_new in rightData["shapes"]:
                        if item_new["type"] == "rect":
                            right_exists = any(
                                ann["name"] == right_name
                                and ann["instrument"] == "Oak1Right"
                                and ann["location"]
                                == [
                                    [item_new["x0"], item_new["y0"]],
                                    [item_new["x1"], item_new["y1"]],
                                ]
                                for ann in annotations
                            )
                            if not right_exists:
                                # return True, no_update, no_update
                                modal_should_open = True
                                break

            if modal_should_open:
                return True, left_fig, right_fig, no_update

            if figure_was_updated:
                return False, left_fig, right_fig, shape_names

            raise PreventUpdate

        @self.app.callback(
            Output("OakLeft_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("OakRight_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("annotations", "data", allow_duplicate=True),
            Output(
                "note_modal_{}".format(self.suffix), "is_open", allow_duplicate=True
            ),
            Output("note_{}".format(self.suffix), "value"),
            State("calibration", "data"),
            State("rectify", "value"),
            State("annotations", "data"),
            State("OakLeft_{}".format(self.suffix), "figure"),
            State("OakRight_{}".format(self.suffix), "figure"),
            State("OakLeft_id_{}".format(self.suffix), "children"),
            State("OakRight_id_{}".format(self.suffix), "children"),
            State("note_{}".format(self.suffix), "value"),
            Input("add_note_{}".format(self.suffix), "n_clicks"),
        )
        def add_note(
            calibration_data,
            rectify,
            annotations,
            left_fig,
            right_fig,
            left_name,
            right_name,
            comment,
            n_clicks,
        ):
            """
            Adds a comment annotation to a rectangular shape drawn on either the
            left or right stereo image when the 'Submit' button in the
            note modal is clicked.

            Args:
                calibration_data: Loaded camera calibration data.
                rectify: Boolean indicating if images are currently rectified.
                annotations: Current list of annotations.
                left_fig, right_fig: Current Plotly figures for left and right images.
                left_name, right_name: File names of the current left and right images.
                comment: The text content of the note.
                n_clicks: Number of times the 'Submit' button was clicked.
            """
            if n_clicks is None:
                raise PreventUpdate
            left_fig = go.Figure(left_fig)
            right_fig = go.Figure(right_fig)

            if annotations is None:
                annotations = []

            if left_fig.layout.shapes:
                for item in left_fig.layout.shapes:
                    if item["type"] == "rect":
                        # print(item)
                        left_ann = {
                            "name": left_name,
                            "instrument": "Oak1Left",
                            "type": "comment",
                            "rectified": rectify,
                            "location": [
                                [item["x0"], item["y0"]],
                                [item["x1"], item["y1"]],
                            ],
                            "values": comment,
                        }
                        left_exists = any(
                            ann["name"] == left_name
                            and ann["instrument"] == "Oak1Left"
                            and ann["location"]
                            == [[item["x0"], item["y0"]], [item["x1"], item["y1"]]]
                            for ann in annotations
                        )
                        if not left_exists:
                            left_fig.add_shape(
                                type="rect",
                                x0=item["x0"],
                                y0=item["y0"],
                                x1=item["x1"],
                                y1=item["y1"],
                                editable=False,
                                line=dict(width=2, color="red"),
                                label=dict(
                                    text=comment,
                                    font=dict(color="red"),
                                    textposition="top left",
                                ),
                                name="annotated_rect_left",
                            )
                            annotations.append(left_ann)
            if right_fig.layout.shapes:
                for item in right_fig.layout.shapes:
                    if item["type"] == "rect":
                        # print(item)
                        right_ann = {
                            "name": right_name,
                            "instrument": "Oak1Right",
                            "type": "comment",
                            "rectified": rectify,
                            "location": [
                                [item["x0"], item["y0"]],
                                [item["x1"], item["y1"]],
                            ],
                            "values": comment,
                        }
                        right_exists = any(
                            ann["name"] == right_name
                            and ann["instrument"] == "Oak1Right"
                            and ann["location"]
                            == [[item["x0"], item["y0"]], [item["x1"], item["y1"]]]
                            for ann in annotations
                        )
                        if not right_exists:
                            right_fig.add_shape(
                                type="rect",
                                x0=item["x0"],
                                y0=item["y0"],
                                x1=item["x1"],
                                y1=item["y1"],
                                editable=False,
                                line=dict(width=2, color="red"),
                                label=dict(
                                    text=comment,
                                    font=dict(color="red"),
                                    textposition="top left",
                                ),
                                name="annotated_rect_right",
                            )
                            annotations.append(right_ann)

            return left_fig, right_fig, annotations, False, ""
