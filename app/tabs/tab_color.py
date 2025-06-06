from dash import Dash, html, dcc, callback, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import dash_extensions as de
import plotly.graph_objects as go

from utils.utils import add_annotations
from typing import List, Tuple, Dict, Any, Optional


class ColorOakTab:
    """
    Represents the 'Color Camera' tab in the Dash application.
    This tab displays images from the center (color) camera.
    """

    def __init__(self, tab_suffix: str, config: Dict[str, Any]):
        """
        Initializes the ColorOakTab.

        Args:
            tab_suffix: A string suffix to make component IDs unique (e.g., "color").
            config: Configuration dictionary for Plotly graphs.
        """
        self.suffix = tab_suffix
        self.config = config

    def create_layout(self) -> html.Div:
        """
        Creates the layout for the 'Color Camera' tab.
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
                        dcc.Store(id="index_{}".format(self.suffix)),
                        dcc.Store(id="alert_state_{}".format(self.suffix)),
                    ],
                    justify="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                id="OakCenter_id_{}".format(self.suffix),
                                style={"font-size": "10px"},
                            )
                        ),
                    ],
                    class_name="g-0",
                    justify="center",
                ),
                dbc.Row(
                    [
                        # html.Div(
                        dcc.Graph(
                            id="OakCenter_{}".format(self.suffix),
                            responsive=True,
                            config=self.config,
                            style={
                                "height": "70vh",
                                "width": "100%",
                                "display": "inline-block",
                            },
                        ),
                        # style={"height": "70vh", "width": "90vw"},
                        # ),
                    ]
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


class ColorOakCallbacks:
    """
    Manages all the callbacks for the 'Color Camera' tab.
    """

    def __init__(self, tab_suffix: str, app: Dash):
        """
        Initializes the ColorOakCallbacks.

        Args:
            tab_suffix: The string suffix used for component IDs in this tab.
            app: The main Dash application instance.
        """
        self.suffix = tab_suffix
        self.app = app

    def create_callbacks(self) -> None:
        """
        Defines and registers all callbacks for the 'Color Camera' tab.
        """

        @self.app.callback(
            Output("OakCenter_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output(
                "OakCenter_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            State("data_paths", "data"),
            State("annotations", "data"),
            Input("load_{}".format(self.suffix), "n_clicks"),
        )
        def display_center(data, annotations, n_clicks):
            """
            Loads and displays the initial image from the center camera
            when the 'Load Data' button is clicked.
            Adds existing annotations to the figure.
            """
            if n_clicks is None or data is None:
                raise PreventUpdate
            else:
                path_data = pd.DataFrame(data)
                img = Image.open(path_data["Oak1Center"].iloc[0])
                fig = px.imshow(img, color_continuous_scale="gray", binary_string=True)
                fig.update_layout(
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, b=0, t=0),
                    autosize=True,
                )
                fig.update_layout(
                    dragmode="drawrect", newshape=dict(line_color="cyan", line_width=1)
                )
                fig.update_xaxes(showticklabels=False)
                fig.update_yaxes(showticklabels=False)
                fig = add_annotations(
                    annotations,
                    fig,
                    "Oak1Center",
                    path_data["Oak1Center"].iloc[0],
                    rectify=False,
                    calib_data=None,
                    img_shape=np.array(img).shape,
                )
                # fig.update_layout(hovermode=False)
                return (
                    fig,
                    path_data["Oak1Center"].iloc[0],
                )

        @self.app.callback(
            Output("OakCenter_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output(
                "OakCenter_id_{}".format(self.suffix), "children", allow_duplicate=True
            ),
            State("data_paths", "data"),
            State("annotations", "data"),
            Input("index_{}".format(self.suffix), "data"),
        )
        def update_center(data, annotations, index):
            """
            Updates the displayed image from the center camera when the image index changes
            (e.g., via Next/Previous buttons). Adds annotations.
            """
            if index is None or data is None:
                raise PreventUpdate
            path_data = pd.DataFrame(data)
            img = Image.open(path_data["Oak1Center"].iloc[index])
            fig = px.imshow(img, color_continuous_scale="gray", binary_string=True)
            fig.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, b=0, t=0),
                autosize=True,
            )
            fig.update_layout(
                dragmode="drawrect", newshape=dict(line_color="cyan", line_width=1)
            )
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            fig = add_annotations(
                annotations,
                fig,
                "Oak1Center",
                path_data["Oak1Center"].iloc[index],
                rectify=False,
                calib_data=None,
                img_shape=np.array(img).shape,
            )
            return fig, path_data["Oak1Center"].iloc[index]

        @self.app.callback(
            Output(
                "note_modal_{}".format(self.suffix), "is_open", allow_duplicate=True
            ),
            State("annotations", "data"),
            State("OakCenter_id_{}".format(self.suffix), "children"),
            Input("OakCenter_{}".format(self.suffix), "relayoutData"),
        )
        def open_modal(annotations, center_name, centerData):
            """
            Opens the note modal if a new rectangular shape (for commenting) is drawn
            on the center image and that shape has not already been annotated.

            Args:
                annotations: Current list of annotations.
                center_name: File name of the current center image.
                centerData: relayoutData from the center image figure.
            """
            if centerData is None:
                raise PreventUpdate
            if "shapes" not in centerData:
                raise PreventUpdate
            if annotations is None:
                annotations = []
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
            Output("OakCenter_{}".format(self.suffix), "figure", allow_duplicate=True),
            Output("annotations", "data", allow_duplicate=True),
            Output(
                "note_modal_{}".format(self.suffix), "is_open", allow_duplicate=True
            ),
            Output("note_{}".format(self.suffix), "value"),
            State("calibration", "data"),
            State("rectify", "value"),
            State("annotations", "data"),
            State("OakCenter_{}".format(self.suffix), "figure"),
            State("OakCenter_id_{}".format(self.suffix), "children"),
            State("note_{}".format(self.suffix), "value"),
            Input("add_note_{}".format(self.suffix), "n_clicks"),
        )
        def add_note(
            calibration_data,
            rectify,
            annotations,
            fig,
            center_name,
            comment,
            n_clicks,
        ):
            """
            Adds a comment annotation to a rectangular shape drawn on the center image
            when the 'Submit' button in the note modal is clicked.

            Args:
                calibration_data: Loaded camera calibration data (unused here).
                rectify: Boolean indicating if images are currently rectified (unused here).
                annotations: Current list of annotations.
                fig: Current Plotly figure for the center image.
                center_name: File name of the current center image.
                comment: The text content of the note.
                n_clicks: Number of times the 'Submit' button was clicked.
            """
            if n_clicks is None:
                raise PreventUpdate

            fig = go.Figure(fig)

            if annotations is None:
                annotations = []

            if fig.layout.shapes:
                for item in fig.layout.shapes:
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
                            fig.add_shape(
                                type="rect",
                                x0=item["x0"],
                                y0=item["y0"],
                                x1=item["x1"],
                                y1=item["y1"],
                                editable=True,
                                line=dict(width=2, color="red"),
                                label=dict(
                                    text=comment,
                                    font=dict(color="red"),
                                    textposition="top left",
                                ),
                            )
                            annotations.append(center_ann)

            return fig, annotations, False, ""
