from dash import Dash, html, dcc, callback, Input, Output, State, no_update
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
    undistort_triangulate_pts,
    calculate_size,
    triangulate_pts,
    add_annotations,
)
from typing import List, Tuple, Dict, Any, Optional, Union


class AllOakTab:
    """
    Represents the 'All Cameras' tab in the Dash application.
    This tab displays images from left, center, and right cameras simultaneously.
    """

    def __init__(self, tab_suffix: str, config: Dict[str, Any]):
        """
        Initializes the AllOakTab.

        Args:
            tab_suffix: A string suffix to make component IDs unique (e.g., "all").
            config: Configuration dictionary for Plotly graphs.
        """
        self.suffix = tab_suffix
        self.config = config

    def create_layout(self) -> html.Div:
        """
        Creates the layout for the 'All Cameras' tab.
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
                                n_clicks=0,
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
                        dbc.Col(
                            html.Div(
                                id="OakCenter_id_{}".format(self.suffix),
                                style={"font-size": "10px"},
                            )
                        ),
                        dbc.Col(
                            html.Div(
                                id="OakRight_id_{}".format(self.suffix),
                                style={"font-size": "10px"},
                            )
                        ),
                        dcc.Store(id="left_shape_{}".format(self.suffix)),
                        dcc.Store(id="right_shape_{}".format(self.suffix)),
                        dcc.Store(id="center_shape_{}".format(self.suffix)),
                    ],
                    class_name="g-0",
                ),
                dbc.Row(
                    [
                        # dbc.Stack(
                        #     [
                        # html.Div(
                        dbc.Col(
                            dcc.Graph(
                                id="OakLeft_{}".format(self.suffix),
                                responsive=True,
                                config=self.config,
                                style={
                                    "width": "100%",
                                    "height": "70vh",
                                },
                            )
                        ),
                        # style={"height": "100%", "width": "100%"},
                        # ),
                        # html.Div(
                        dbc.Col(
                            dcc.Graph(
                                id="OakCenter_{}".format(self.suffix),
                                responsive=True,
                                config=self.config,
                                style={
                                    "width": "100%",
                                    "height": "70vh",
                                },
                            )
                        ),
                        # style={"height": "100%", "width": "100%"},
                        # ),
                        # html.Div(
                        dbc.Col(
                            dcc.Graph(
                                id="OakRight_{}".format(self.suffix),
                                responsive=True,
                                config=self.config,
                                style={
                                    "width": "100%",
                                    "height": "65vh",
                                },
                            )
                        ),
                        # style={"height": "100%", "width": "100%"},
                        # ),
                        # ],
                        # direction="horizontal",
                        # ),
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
                                    ]
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


class AllOakCallbacks:
    """
    Manages all the callbacks for the 'All Cameras' tab.
    """

    def __init__(self, tab_suffix: str, app: Dash):
        """
        Initializes the AllOakCallbacks.

        Args:
            tab_suffix: The string suffix used for component IDs in this tab.
            app: The main Dash application instance.
        """
        self.suffix = tab_suffix
        self.app = app

    def create_callbacks(self) -> None:
        """
        Defines and registers all callbacks for the 'All Cameras' tab.
        """

        @self.app.callback(
            Output("OakLeft_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("OakCenter_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("OakRight_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output(
                "OakLeft_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output(
                "OakCenter_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output(
                "OakRight_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output("left_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            Output("right_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            Output("center_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            State("data_paths", "data"),
            State("calibration", "data"),
            State("annotations", "data"),
            State("rectify", "value"),
            Input("load_{}".format(self.suffix), "n_clicks"),
        )
        def display_all(data, calibration_data, annotations, rectify, n_clicks):
            """
            Loads and displays the initial set of images (left, center, right)
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
                c_times = pd.to_datetime(path_data["Oak1CenterTimes"])
                if abs((c_times.iloc[0] - l_times.iloc[0]).total_seconds()) < 0.05:
                    img2 = Image.open(path_data["Oak1Center"].iloc[0])
                else:
                    img2 = Image.open("assets/MS_Full_White.png")
                if abs((r_times.iloc[0] - l_times.iloc[0]).total_seconds()) < 0.05:
                    img3 = Image.open(path_data["Oak1Right"].iloc[0])
                else:
                    img3 = Image.open("assets/MS_Full_White.png")

                if rectify:
                    if calibration_data is not None:
                        img1 = undistortrectify(
                            np.array(img1), calibration_data, "left"
                        )
                        # img2 = undistortrectify(np.array(img2), calibration_data, "center")
                        img3 = undistortrectify(
                            np.array(img3), calibration_data, "right"
                        )

                fig1 = px.imshow(
                    img1, color_continuous_scale="gray", binary_string=True
                )
                fig1.update_layout(
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, b=0, t=0),
                    autosize=True,
                    shapes=[],  # Clear existing shapes
                    annotations=[],  # Clear existing annotations
                )
                fig1.update_layout(
                    dragmode="drawline", newshape=dict(line_color="cyan", line_width=1)
                )
                fig1.update_xaxes(showticklabels=False)
                fig1.update_yaxes(showticklabels=False)

                fig2 = px.imshow(
                    img2, color_continuous_scale="gray", binary_string=True
                )
                fig2.update_layout(
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, b=0, t=0),
                    autosize=True,
                    shapes=[],  # Clear existing shapes
                    annotations=[],  # Clear existing annotations
                )
                fig2.update_layout(
                    dragmode="drawline", newshape=dict(line_color="cyan", line_width=1)
                )
                fig2.update_xaxes(showticklabels=False)
                fig2.update_yaxes(showticklabels=False)

                fig3 = px.imshow(
                    img3, color_continuous_scale="gray", binary_string=True
                )
                fig3.update_layout(
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, b=0, t=0),
                    autosize=True,
                    shapes=[],  # Clear existing shapes
                    annotations=[],  # Clear existing annotations
                )
                fig3.update_layout(
                    dragmode="drawline", newshape=dict(line_color="cyan", line_width=1)
                )
                fig3.update_xaxes(showticklabels=False)
                fig3.update_yaxes(showticklabels=False)
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
                    "Oak1Center",
                    path_data["Oak1Center"].iloc[0],
                    False,
                    calibration_data,
                    np.array(img2).shape,
                )
                fig3 = add_annotations(
                    annotations,
                    fig3,
                    "Oak1Right",
                    path_data["Oak1Right"].iloc[0],
                    rectify,
                    calibration_data,
                    np.array(img3).shape,
                )
                return (
                    fig1,
                    fig2,
                    fig3,
                    path_data["Oak1Left"].iloc[0],
                    path_data["Oak1Center"].iloc[0],
                    path_data["Oak1Right"].iloc[0],
                    np.array(img1).shape,
                    np.array(img3).shape,
                    np.array(img2).shape,
                )

        @self.app.callback(
            Output("OakLeft_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("OakCenter_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("OakRight_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output(
                "OakLeft_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output(
                "OakCenter_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output(
                "OakRight_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output("left_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            Output("right_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            Output("center_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            State("data_paths", "data"),
            State("calibration", "data"),
            State("annotations", "data"),
            State("rectify", "value"),
            Input("index_{}".format(self.suffix), "data"),
        )
        def update_all(data, calibration_data, annotations, rectify, index):
            """
            Updates the displayed images (left, center, right) when the image index changes
            (e.g., via Next/Previous buttons).
            Handles image rectification and adds annotations.
            """
            if index is None or data is None:
                raise PreventUpdate
            path_data = pd.DataFrame(data)
            img1 = Image.open(path_data["Oak1Left"].iloc[index])
            r_times = pd.to_datetime(path_data["Oak1RightTimes"])
            l_times = pd.to_datetime(path_data["Oak1LeftTimes"])
            c_times = pd.to_datetime(path_data["Oak1CenterTimes"])
            idx1 = np.argmin(abs(l_times.iloc[index] - r_times))
            idx2 = np.argmin(abs(l_times.iloc[index] - c_times))
            if abs((l_times.iloc[index] - c_times.iloc[idx2]).total_seconds()) > 0.05:
                img2 = Image.open("assets/MS_Full_White.png")
            else:
                img2 = Image.open(path_data["Oak1Center"].iloc[idx2])
            if abs((l_times.iloc[index] - r_times.iloc[idx1]).total_seconds()) > 0.05:
                img3 = Image.open("assets/MS_Full_White.png")
            else:
                img3 = Image.open(path_data["Oak1Right"].iloc[idx1])

            if rectify:
                if calibration_data is not None:
                    img1 = undistortrectify(np.array(img1), calibration_data, "left")
                    # img2 = undistortrectify(np.array(img2), calibration_data, "center")
                    img3 = undistortrectify(np.array(img3), calibration_data, "right")
            fig1 = px.imshow(img1, color_continuous_scale="gray", binary_string=True)
            fig1.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, b=0, t=0),
                autosize=True,
                shapes=[],  # Clear existing shapes
                annotations=[],  # Clear existing annotations
            )
            fig1.update_layout(
                dragmode="drawline", newshape=dict(line_color="cyan", line_width=1)
            )
            fig1.update_xaxes(showticklabels=False)
            fig1.update_yaxes(showticklabels=False)

            fig2 = px.imshow(img2, color_continuous_scale="gray", binary_string=True)
            fig2.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, b=0, t=0),
                autosize=True,
                shapes=[],  # Clear existing shapes
                annotations=[],  # Clear existing annotations
            )
            fig2.update_layout(
                dragmode="drawline", newshape=dict(line_color="cyan", line_width=1)
            )
            fig2.update_xaxes(showticklabels=False)
            fig2.update_yaxes(showticklabels=False)

            fig3 = px.imshow(img3, color_continuous_scale="gray", binary_string=True)
            fig3.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, b=0, t=0),
                autosize=True,
                shapes=[],  # Clear existing shapes
                annotations=[],  # Clear existing annotations
            )
            fig3.update_layout(
                dragmode="drawline", newshape=dict(line_color="cyan", line_width=1)
            )
            fig3.update_xaxes(showticklabels=False)
            fig3.update_yaxes(showticklabels=False)
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
                "Oak1Center",
                path_data["Oak1Center"].iloc[idx2],
                False,
                calibration_data,
                np.array(img2).shape,
            )
            fig3 = add_annotations(
                annotations,
                fig3,
                "Oak1Right",
                path_data["Oak1Right"].iloc[idx1],
                rectify,
                calibration_data,
                np.array(img3).shape,
            )
            return (
                fig1,
                fig2,
                fig3,
                path_data["Oak1Left"].iloc[index],
                path_data["Oak1Center"].iloc[idx2],
                path_data["Oak1Right"].iloc[idx1],
                np.array(img1).shape,
                np.array(img3).shape,
                np.array(img2).shape,
            )

        @self.app.callback(
            Output("OakLeft_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("OakCenter_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("OakRight_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output(
                "OakLeft_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output(
                "OakCenter_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output(
                "OakRight_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            Output("alert_state_{}".format(self.suffix), "data", allow_duplicate=True),
            Output("left_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            Output("right_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            Output("center_shape_{}".format(self.suffix), "data", allow_duplicate=True),
            State("data_paths", "data"),
            State("calibration", "data"),
            State("annotations", "data"),
            State("index_{}".format(self.suffix), "data"),
            Input("rectify", "value"),
            prevent_initial_call=True,
        )
        def rectify_all(data, calibration_data, annotations, index, rectify):
            """
            Updates the displayed images (left, center, right) when the 'Rectify' switch changes.
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
            c_times = pd.to_datetime(path_data["Oak1CenterTimes"])
            idx1 = np.argmin(abs(l_times.iloc[index] - r_times))
            idx2 = np.argmin(abs(l_times.iloc[index] - c_times))
            if abs((l_times.iloc[index] - c_times.iloc[idx2]).total_seconds()) > 0.05:
                img2 = Image.open("assets/MS_Full_White.png")
            else:
                img2 = Image.open(path_data["Oak1Center"].iloc[idx2])
            if abs((l_times.iloc[index] - r_times.iloc[idx1]).total_seconds()) > 0.05:
                img3 = Image.open("assets/MS_Full_White.png")
            else:
                img3 = Image.open(path_data["Oak1Right"].iloc[idx1])
            if rectify:
                if calibration_data is not None:
                    img1 = undistortrectify(np.array(img1), calibration_data, "left")
                    # img2 = undistortrectify(np.array(img2), calibration_data, "center")
                    img3 = undistortrectify(np.array(img3), calibration_data, "right")
            fig1 = px.imshow(img1, color_continuous_scale="gray", binary_string=True)
            fig1.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, b=0, t=0),
                autosize=True,
                shapes=[],  # Clear existing shapes
                annotations=[],  # Clear existing annotations
            )
            fig1.update_layout(
                dragmode="drawline", newshape=dict(line_color="cyan", line_width=1)
            )
            fig1.update_xaxes(showticklabels=False)
            fig1.update_yaxes(showticklabels=False)

            fig2 = px.imshow(img2, color_continuous_scale="gray", binary_string=True)
            fig2.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, b=0, t=0),
                autosize=True,
                shapes=[],  # Clear existing shapes
                annotations=[],  # Clear existing annotations
            )
            fig2.update_layout(
                dragmode="drawline", newshape=dict(line_color="cyan", line_width=1)
            )
            fig2.update_xaxes(showticklabels=False)
            fig2.update_yaxes(showticklabels=False)

            fig3 = px.imshow(img3, color_continuous_scale="gray", binary_string=True)
            fig3.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, b=0, t=0),
                autosize=True,
                shapes=[],  # Clear existing shapes
                annotations=[],  # Clear existing annotations
            )
            fig3.update_layout(
                dragmode="drawline", newshape=dict(line_color="cyan", line_width=1)
            )
            fig3.update_xaxes(showticklabels=False)
            fig3.update_yaxes(showticklabels=False)
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
                "Oak1Center",
                path_data["Oak1Center"].iloc[idx2],
                False,
                calibration_data,
                np.array(img2).shape,
            )
            fig3 = add_annotations(
                annotations,
                fig3,
                "Oak1Right",
                path_data["Oak1Right"].iloc[idx1],
                rectify,
                calibration_data,
                np.array(img3).shape,
            )
            if calibration_data is None:
                return (
                    fig1,
                    fig2,
                    fig3,
                    path_data["Oak1Left"].iloc[index],
                    path_data["Oak1Center"].iloc[idx2],
                    path_data["Oak1Right"].iloc[idx1],
                    {
                        "state": True,
                        "msg": "Looks like you're missing a calibration file. Upload one using the selector above.",
                    },
                    np.array(img1).shape,
                    np.array(img3).shape,
                    np.array(img2).shape,
                )
            else:
                return (
                    fig1,
                    fig2,
                    fig3,
                    path_data["Oak1Left"].iloc[index],
                    path_data["Oak1Center"].iloc[idx2],
                    path_data["Oak1Right"].iloc[idx1],
                    no_update,
                    np.array(img1).shape,
                    np.array(img3).shape,
                    np.array(img2).shape,
                )

        @self.app.callback(
            Output("OakLeft_{}".format(self.suffix), "figure"),
            Output("OakRight_{}".format(self.suffix), "figure"),
            Output("annotations", "data", allow_duplicate=True),
            Output("alert_state_{}".format(self.suffix), "data", allow_duplicate=True),
            State("calibration", "data"),
            State("rectify", "value"),
            State("annotations", "data"),
            State("OakLeft_{}".format(self.suffix), "figure"),
            State("OakRight_{}".format(self.suffix), "figure"),
            State("OakLeft_id_{}".format(self.suffix), "children"),
            State("OakRight_id_{}".format(self.suffix), "children"),
            State("OakLeft_{}".format(self.suffix), "relayoutData"),
            State("OakRight_{}".format(self.suffix), "relayoutData"),
            State("left_shape_{}".format(self.suffix), "data"),
            State("right_shape_{}".format(self.suffix), "data"),
            State("center_shape_{}".format(self.suffix), "data"),
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
            leftData,
            rightData,
            left_shape,
            right_shape,
            center_shape,
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
                center_shape: Shape of the current center image (unused in this function).
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
            if leftData is None and rightData is None:
                raise PreventUpdate
            if (
                "shapes" not in leftData
                or "shapes" not in rightData
                or not leftData["shapes"]
                or not rightData["shapes"]
            ):
                raise PreventUpdate

            left_pts = []
            right_pts = []
            left_count = 0
            right_count = 0

            left_fig = go.Figure(left_fig)
            right_fig = go.Figure(right_fig)

            if annotations is None:
                annotations = []

            for item in leftData["shapes"]:
                if item["type"] == "line":
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

            for item in rightData["shapes"]:
                if item["type"] == "line":
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
                raise PreventUpdate

            if rectify:
                pts = triangulate_pts(calibration_data, left_pts, right_pts, left_shape)
            else:
                pts = undistort_triangulate_pts(
                    calibration_data, left_pts, right_pts, left_shape
                )
            distances = calculate_size(pts)

            # Clear existing shapes before adding new ones
            left_fig.update_layout(shapes=[])
            right_fig.update_layout(shapes=[])

            for i, distance in enumerate(distances):
                left_ann = {
                    "name": left_name,
                    "instrument": "Oak1Left",
                    "type": "distance",
                    "rectified": rectify,
                    "location": left_pts[i * 2 : i * 2 + 2],
                    "values": distances[i],
                }

                right_ann = {
                    "name": right_name,
                    "instrument": "Oak1Right",
                    "type": "distance",
                    "rectified": rectify,
                    "location": right_pts[i * 2 : i * 2 + 2],
                    "values": distances[i],
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
                        x0=left_pts[i * 2][0],
                        y0=left_pts[i * 2][1],
                        x1=left_pts[i * 2 + 1][0],
                        y1=left_pts[i * 2 + 1][1],
                        line_width=3,
                        line_color="red",
                        label=dict(
                            text=str(distances[i]) + " m", font=dict(color="red")
                        ),
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
                        x0=right_pts[i * 2][0],
                        y0=right_pts[i * 2][1],
                        x1=right_pts[i * 2 + 1][0],
                        y1=right_pts[i * 2 + 1][1],
                        line_width=3,
                        line_color="red",
                        label=dict(
                            text=str(distances[i]) + " m", font=dict(color="red")
                        ),
                    )
            return left_fig, right_fig, annotations, no_update

        @self.app.callback(
            Output(
                "note_modal_{}".format(self.suffix), "is_open", allow_duplicate=True
            ),
            State("annotations", "data"),
            State("OakLeft_id_{}".format(self.suffix), "children"),
            State("OakCenter_id_{}".format(self.suffix), "children"),
            State("OakRight_id_{}".format(self.suffix), "children"),
            Input("OakLeft_{}".format(self.suffix), "relayoutData"),
            Input("OakCenter_{}".format(self.suffix), "relayoutData"),
            Input("OakRight_{}".format(self.suffix), "relayoutData"),
        )
        def open_modal(
            annotations,
            left_name,
            center_name,
            right_name,
            leftData,
            centerData,
            rightData,
        ):
            """
            Opens the note modal if a new rectangular shape (for commenting) is drawn
            on any of the three images (left, center, right) and that shape
            has not already been annotated.

            Args:
                annotations: Current list of annotations.
                left_name, center_name, right_name: File names of the current images.
                leftData, centerData, rightData: relayoutData from the figures.

            Returns:
                True to open the modal, otherwise raises PreventUpdate.
            """
            if leftData is None and rightData is None and centerData is None:
                raise PreventUpdate
            if (
                "shapes" not in leftData
                and "shapes" not in rightData
                and not "shapes" in centerData
            ):
                raise PreventUpdate
            if annotations is None:
                annotations = []
            if "shapes" in leftData:
                for item in leftData["shapes"]:
                    if item["type"] == "rect":
                        left_exists = any(
                            ann["name"] == left_name
                            and ann["instrument"] == "Oak1Left"
                            and ann["location"]
                            == [[item["x0"], item["y0"]], [item["x1"], item["y1"]]]
                            for ann in annotations
                        )
                        if not left_exists:
                            return True
            if "shapes" in rightData:
                for item in rightData["shapes"]:
                    if item["type"] == "rect":
                        right_exists = any(
                            ann["name"] == right_name
                            and ann["instrument"] == "Oak1Right"
                            and ann["location"]
                            == [[item["x0"], item["y0"]], [item["x1"], item["y1"]]]
                            for ann in annotations
                        )
                        if not right_exists:
                            return True
            if "shapes" in centerData:
                for item in centerData["shapes"]:
                    if item["type"] == "rect":
                        center_exists = any(
                            ann["name"] == center_name
                            and ann["instrument"] == "Oak1Center"
                            and ann["location"]
                            == [[item["x0"], item["y0"]], [item["x1"], item["y1"]]]
                            for ann in annotations
                        )
                        if not center_exists:
                            return True
            raise PreventUpdate

        @self.app.callback(
            Output("OakLeft_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("OakCenter_{}".format(self.suffix), "figure", allow_duplicate=True),
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
            State("OakCenter_{}".format(self.suffix), "figure"),
            State("OakRight_{}".format(self.suffix), "figure"),
            State("OakLeft_id_{}".format(self.suffix), "children"),
            State("OakCenter_id_{}".format(self.suffix), "children"),
            State("OakRight_id_{}".format(self.suffix), "children"),
            State("note_{}".format(self.suffix), "value"),
            Input("add_note_{}".format(self.suffix), "n_clicks"),
        )
        def add_note(
            calibration_data,
            rectify,
            annotations,
            left_fig,
            center_fig,
            right_fig,
            left_name,
            center_name,
            right_name,
            comment,
            n_clicks,
        ):
            """
            Adds a comment annotation to a rectangular shape drawn on any of the
            three images (left, center, right) when the 'Submit' button in the
            note modal is clicked.

            Args:
                calibration_data: Loaded camera calibration data.
                rectify: Boolean indicating if images are currently rectified.
                annotations: Current list of annotations.
                left_fig, center_fig, right_fig: Current Plotly figures.
                left_name, center_name, right_name: File names of the current images.
                comment: The text content of the note.
                n_clicks: Number of times the 'Submit' button was clicked.
            """

            if n_clicks is None:
                raise PreventUpdate
            left_fig = go.Figure(left_fig)
            center_fig = go.Figure(center_fig)
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
                                line=dict(width=2, color="red"),
                                label=dict(
                                    text=comment,
                                    font=dict(color="red"),
                                    textposition="top left",
                                ),
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
                                line=dict(width=2, color="red"),
                                label=dict(
                                    text=comment,
                                    font=dict(color="red"),
                                    textposition="top left",
                                ),
                            )
                            annotations.append(right_ann)

            if center_fig.layout.shapes:
                for item in center_fig.layout.shapes:
                    if item["type"] == "rect":
                        # print(item)
                        center_ann = {
                            "name": center_name,
                            "instrument": "Oak1Center",
                            "type": "comment",
                            "rectified": False,
                            "location": [
                                [item["x0"], item["y0"]],
                                [item["x1"], item["y1"]],
                            ],
                            "values": comment,
                        }
                        center_exists = any(
                            ann["name"] == center_name
                            and ann["instrument"] == "Oak1Center"
                            and ann["location"]
                            == [[item["x0"], item["y0"]], [item["x1"], item["y1"]]]
                            for ann in annotations
                        )
                        if not center_exists:
                            center_fig.add_shape(
                                type="rect",
                                x0=item["x0"],
                                y0=item["y0"],
                                x1=item["x1"],
                                y1=item["y1"],
                                line=dict(width=2, color="red"),
                                label=dict(
                                    text=comment,
                                    font=dict(color="red"),
                                    textposition="top left",
                                ),
                            )
                            annotations.append(center_ann)

            return left_fig, center_fig, right_fig, annotations, False, ""
