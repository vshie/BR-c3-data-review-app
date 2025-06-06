from dash import Dash, html, dcc, callback, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import os
import numpy as np
import pandas as pd
import io
from PIL import Image, ExifTags
import plotly.express as px
import logging
from glob import glob
import platform
import datetime

# import classes for tabs
from tabs.tab_stereo import StereoOakTab, StereoOakCallbacks
from tabs.tab_color import ColorOakTab, ColorOakCallbacks
from tabs.tab_all import AllOakTab, AllOakCallbacks
from tabs.universal_callbacks import UniversalCallbacks

import json
import base64
from typing import List, Dict, Tuple, Any, Optional, Union
from waitress import serve


logger = logging.getLogger("__main__")
logger.setLevel(logging.INFO)


MOUNT_POINT = "/mnt/data"

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    dbc.icons.BOOTSTRAP,
    "/assets/custom.css",
    "https://fonts.googleapis.com/css2?family=Catamaran&display=swap",
    "https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap",
    "https://fonts.googleapis.com/css2?family=Biryani:wght@200;300;400;600;700;800;900&display=swap",
]

app = Dash(
    external_stylesheets=external_stylesheets,
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=True,
    title="C3 Data Review App",
)

app.css.config.serve_locally = True

config = {
    "modeBarButtonsToAdd": [
        "drawline",
        "drawrect",
        "eraseshape",
    ],
    "responsive": True,
    "scrollZoom": True,
}

#   Initialize tabs
tabs = ["stereo", "color", "all"]
stereo_interface = StereoOakTab("stereo", config)
color_interface = ColorOakTab("color", config)
all_interface = AllOakTab("all", config)

stereo_callbacks = StereoOakCallbacks("stereo", app)
color_callbacks = ColorOakCallbacks("color", app)
all_callbacks = AllOakCallbacks("all", app)
univ = UniversalCallbacks(tabs, app)


def get_times(files: List[str]) -> List[Dict[str, str]]:
    """
    Extracts creation times from image files using EXIF data.

    Args:
        files: A list of file paths to the images.

    Returns:
        A list of dictionaries, where each dictionary contains the 'time' (as a string)
        and 'file' path. The list is sorted by time.
    """
    time_list = []
    for file in files:
        try:
            img = Image.open(file)
        except Exception as e:
            logger.error("Malformed image: %s", e)
            continue
        if img.getexif() is not None:
            try:
                img_time = img.getexif()[306]
                time_list.append(
                    {
                        "time": datetime.datetime.strptime(
                            img_time, "%Y-%m-%d %H:%M:%S.%f"
                        ),
                        "file": file,
                    }
                )
            except Exception as e:
                logger.warning(
                    "No creation time found in exif data. Trying to parse timestamp from filepath."
                )
                try:
                    img_time = datetime.datetime.strptime(
                        file.replace("\\", "/")
                        .split("/")[-1]
                        .replace(".jpg", "")
                        .replace("snapshot-", ""),
                        "%Y-%m-%d-%H-%M-%S-%f",
                    )
                    time_list.append(
                        {
                            "time": img_time,
                            "file": file,
                        }
                    )
                except Exception as e:
                    logger.warning("Unknown time format in filepath. Skipping image.")
                    continue

    time_list.sort(key=lambda x: x["time"])
    for i, time in enumerate(time_list):
        time_list[i]["time"] = time["time"].strftime("%Y-%m-%d %H:%M:%S.%f")
    return time_list


def get_times_vision(files: List[str]) -> List[Dict[str, str]]:
    """
    Extracts creation times from image file names (specific format).

    Args:
        files: A list of file paths to the images.

    Returns:
        A list of dictionaries, where each dictionary contains the 'time' (as a string)
        and 'file' path. The list is sorted by time.
    """

    time_list = []
    for file in files:
        try:
            file_time = datetime.datetime.strptime(
                file.replace("\\", "/").split("/")[-1].replace(".jpg", ""),
                "%Y_%m_%d_%H_%M_%S.%f",
            )
        except Exception as e:
            logger.error("Unknown encoded datetime format: %s", e)
            continue
        time_list.append({"time": file_time, "file": file})
    time_list.sort(key=lambda x: x["time"])
    for i, time in enumerate(time_list):
        time_list[i]["time"] = time["time"].strftime("%Y-%m-%d %H:%M:%S.%f")
    return time_list


def resolve_path(user_input_path: str) -> str:
    """
    Resolves a user-provided path to an absolute path within the mount point.

    Args:
        user_input_path: The path provided by the user.

    Returns:
        The absolute path within the application's mount point.
    """
    relative_path = os.path.relpath(user_input_path, "/app")
    relative_path = relative_path.replace("\\", "/")
    return os.path.join(MOUNT_POINT, relative_path)


def parse_time_from_files(directory: str) -> Tuple[pd.DataFrame, int]:
    """
    Parses image files from specified subdirectories, extracts timestamps,
    and synchronizes them into a Pandas DataFrame.

    Args:
        directory: The root directory containing the following image subdirectories
                   (e.g., 'Oak1Left/', 'Oak1Right/', 'Oak1Center/' or 'left/', 'right/', 'center/').

    Returns:
        A tuple containing:
            - path_data (pd.DataFrame): DataFrame with synchronized image paths and timestamps.
            - num_files (int): Total number of image files found.
    """
    if os.path.exists(os.path.join(directory, "Oak1Left/")):
        stereo1_imgs = glob(os.path.join(directory, "Oak1Left/**/*"), recursive=True)
        stereo2_imgs = glob(os.path.join(directory, "Oak1Right/**/*"), recursive=True)
        color_imgs = glob(os.path.join(directory, "Oak1Center/**/*"), recursive=True)
        stereo1_times = get_times_vision(stereo1_imgs)
        stereo2_times = get_times_vision(stereo2_imgs)
        color_times = get_times_vision(color_imgs)
    elif os.path.exists(os.path.join(directory, "left/")):
        stereo1_imgs = glob(os.path.join(directory, "left/*"))
        stereo2_imgs = glob(os.path.join(directory, "right/*"))
        color_imgs = glob(os.path.join(directory, "center/*"))
        stereo1_times = get_times(stereo1_imgs)
        stereo2_times = get_times(stereo2_imgs)
        color_times = get_times(color_imgs)
    else:
        logger.error("Unknown directory structure")
        return None, 0
    num_files = len(stereo1_times) + len(stereo2_times) + len(color_times)
    if all(
        len(x) == len(stereo1_times)
        for x in [stereo1_times, stereo2_times, color_times]
    ):
        path_data = pd.DataFrame(
            data={
                "Oak1Left": [item["file"] for item in stereo1_times],
                "Oak1LeftTimes": pd.to_datetime(
                    [item["time"] for item in stereo1_times], errors="coerce"
                ),
                "Oak1Right": [item["file"] for item in stereo2_times],
                "Oak1RightTimes": pd.to_datetime(
                    [item["time"] for item in stereo2_times], errors="coerce"
                ),
                "Oak1Center": [item["file"] for item in color_times],
                "Oak1CenterTimes": pd.to_datetime(
                    [item["time"] for item in color_times], errors="coerce"
                ),
            },
        )
    else:
        max_len = max([len(stereo1_times), len(stereo2_times), len(color_times)])
        color_times_new = np.full(max_len, np.nan, dtype=object)
        color_times_new[: len(color_times)] = [item["time"] for item in color_times]
        color_imgs_new = np.full(max_len, "", dtype=object)
        color_imgs_new[: len(color_times)] = [item["file"] for item in color_times]
        stereo1_times_new = np.full(max_len, np.nan, dtype=object)
        stereo1_times_new[: len(stereo1_times)] = [
            item["time"] for item in stereo1_times
        ]
        stereo1_imgs_new = np.full(max_len, "", dtype=object)
        stereo1_imgs_new[: len(stereo1_times)] = [
            item["file"] for item in stereo1_times
        ]
        stereo2_times_new = np.full(max_len, np.nan, dtype=object)
        stereo2_times_new[: len(stereo2_times)] = [
            item["time"] for item in stereo2_times
        ]
        stereo2_imgs_new = np.full(max_len, "", dtype=object)
        stereo2_imgs_new[: len(stereo2_times)] = [
            item["file"] for item in stereo2_times
        ]
        path_data = pd.DataFrame(
            data={
                "Oak1Left": stereo1_imgs_new,
                "Oak1LeftTimes": pd.to_datetime(stereo1_times_new, errors="coerce"),
                "Oak1Right": stereo2_imgs_new,
                "Oak1RightTimes": pd.to_datetime(stereo2_times_new, errors="coerce"),
                "Oak1Center": color_imgs_new,
                "Oak1CenterTimes": pd.to_datetime(color_times_new, errors="coerce"),
            },
        )
    return path_data, num_files


app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Img(
                            src="/assets/MS_Full_White.png", style={"width": "200px"}
                        )
                    ],
                    width="auto",
                ),
                dbc.Col(
                    [
                        html.H2(
                            id="website-title",
                            children="C3 Data Review App",
                            # style={"color": "white", "width": "18vw"},
                            style={"color": "white", "width": "350px"},
                        )
                    ],
                    width="auto",
                    align="center",
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            dbc.Input(
                                placeholder="<path>/<to>/<data>/",
                                style={"width": "350px"},
                                id="filepath",
                            )
                        ),
                        dbc.Row(
                            dcc.Loading(
                                type="circle",
                                color="#0d3151",
                                fullscreen=True,
                                children=[
                                    html.Div(
                                        id="conf-file",
                                        style={
                                            "font-size": "10px",
                                            "color": "white",
                                            "width": "345px",
                                        },
                                    )
                                ],
                                style={"background": "transparent"},
                            ),
                        ),
                    ],
                    width="auto",
                    align="center",
                ),
                dbc.Col(
                    [
                        dbc.Button(
                            # children=["Search"],
                            children=[
                                html.I(
                                    className="bi bi-search",
                                    style={"color": "#f0a120"},
                                ),
                                " Search",
                            ],
                            # style={"width": "5vw"},
                            id="submit",
                            class_name="button",
                        )
                    ],
                    width="auto",
                    align="center",
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            dcc.Upload(
                                id="upload-data",
                                children=html.Div(["  Upload Calibration  "]),
                                style={
                                    "width": "200px",
                                    "height": "50px",
                                    "lineHeight": "50px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "color": "#f0a120",
                                    "border-color": "#f0a120",
                                },
                            )
                        ),
                        dcc.Store(id="calibration"),
                        dbc.Row(
                            html.Div(
                                id="confirm-calib-upload",
                                style={
                                    "font-size": "10px",
                                    "color": "white",
                                    "width": "200px",
                                    # "width": "12vw",
                                },
                            )
                        ),
                    ],
                    width="auto",
                    align="center",
                ),
                dbc.Col(
                    [
                        # dbc.Card(
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Label(
                                        "Rectify Images:",  # style={"width": "7vw"}
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Switch(
                                        id="rectify",
                                        value=False,
                                        # input_style={"borderColor": "#4499A3"},
                                        style={
                                            "fontSize": "20",
                                        },
                                        class_name="gamboge-switch",
                                    ),
                                    width="auto",
                                ),
                            ],
                            align="center",
                            style={
                                "height": "5vh",
                                "color": "white",
                                "textAlign": "center",
                            },
                        ),
                    ],
                    width="auto",
                    align="center",
                ),
                dbc.Col(
                    [
                        dbc.Button(
                            [
                                html.I(
                                    className="bi bi-download",
                                    style={"color": "#f0a120"},
                                ),
                                " Download Notes",
                            ],
                            id="annotation_download",
                            n_clicks=0,
                            class_name="button",
                            # style={"width": "14vw"},
                        ),
                        dcc.Download(id="download"),
                    ],
                    width="auto",
                    align="center",
                ),
            ],
            style={
                "background-color": "#014a6e",
            },
            justify="between",
            class_name="g-0",
        ),
        dbc.Row(
            [
                dbc.Alert(
                    id="alert",
                    color="warning",
                    is_open=False,
                    dismissable=True,
                    duration=5000,
                ),
                dcc.Store(id="alert_state"),
            ],
            style={"background-color": "#014a6e"},
        ),
        dbc.Row(
            [
                dcc.Tabs(
                    id="tabs",
                    value=None,
                    children=[
                        dcc.Tab(
                            label="Stereo",
                            value="stereo",
                            className="tabs",
                            selected_className="selected-tabs",
                            children=stereo_interface.create_layout(),
                        ),
                        dcc.Tab(
                            label="Color",
                            value="color",
                            className="tabs",
                            selected_className="selected-tabs",
                            children=color_interface.create_layout(),
                        ),
                        dcc.Tab(
                            label="All Cameras",
                            value="all-cameras",
                            className="tabs",
                            selected_className="selected-tabs",
                            children=all_interface.create_layout(),
                        ),
                    ],
                ),
            ],
            style={"background-color": "#014a6e"},
        ),
        dcc.Store(id="data_paths"),
        dcc.Store(id="annotations"),
    ],
    fluid=True,
    style={"background-color": "#014a6e"},
)


# Global Callbacks


@app.callback(
    Output("data_paths", "data"),
    Output("calibration", "data", allow_duplicate=True),
    Output("conf-file", "children"),
    State("filepath", "value"),
    Input("submit", "n_clicks"),
)
def get_data(filepath, n_clicks):
    """
    Loads image data and calibration data from the specified filepath when the submit button is clicked.

    Args:
        filepath: The path to the data directory.
        n_clicks: The number of times the submit button has been clicked.

    Returns:
        A tuple containing:
            - path_data (Dict[str, Any]): A dictionary representation of the DataFrame containing image paths and times.
            - calibration_data (Any): The loaded calibration data (JSON content) or `no_update`.
            - confirmation_message (str): A message indicating the number of files found.
    """
    if n_clicks is None:
        raise PreventUpdate
    else:
        filepath = resolve_path(filepath)
        path_data, num_files = parse_time_from_files(filepath)
        if os.path.exists(filepath):
            for filename in os.listdir(filepath):
                # This should specifically point to the name we give the calibrations that we save
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(filepath, filename), "r") as f:
                            calibration_data = json.load(f)
                    except Exception as e:
                        logger.error("Error reading calibration file: %s", e)
                        return (
                            path_data.to_dict(),
                            no_update,
                            f"Found {num_files} files.",
                        )
                    return (
                        path_data.to_dict(),
                        calibration_data,
                        f"Found {num_files} files.",
                    )
                else:
                    continue
            return path_data.to_dict(), no_update, f"Found {num_files} files."
        else:
            return None, no_update, "No files found."


# Initialize tab callbacks and shared callbacks
stereo_callbacks.create_callbacks()
color_callbacks.create_callbacks()
univ.create_callbacks()
all_callbacks.create_callbacks()


@app.callback(
    Output("confirm-calib-upload", "children"),
    Output("calibration", "data", allow_duplicate=True),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def upload_calibration(contents, filename):
    """
    Handles the upload of a calibration file.

    Args:
        contents: The base64 encoded string of the uploaded file's content.
        filename: The name of the uploaded file.

    Returns:
        A tuple containing:
            - confirmation_message (str): A message confirming the upload.
            - decoded_json (Optional[Dict[str, Any]]): The decoded JSON content of the calibration file.
    """
    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = json.loads(base64.b64decode(content_string))
        logger.info("Uploaded calibration file: %s", filename)
        return "Uploaded calibration file: " + filename, decoded


@app.callback(
    Output("download", "data"),
    Input("annotation_download", "n_clicks"),
    State("annotations", "data"),
    prevent_initial_call=True,
)
def download_annotations(n_clicks, annotations):
    """
    Downloads the current annotations as a CSV file.

    Args:
        n_clicks: The number of times the download button has been clicked.
        annotations: A list of annotation dictionaries.

    Returns:
        A Dash `dcc.send_data_frame` object to trigger the file download,
        or raises `PreventUpdate` if no click or annotations.
    """
    if n_clicks is None:
        raise PreventUpdate
    annotations_df = pd.DataFrame(annotations)
    df_final = annotations_df.groupby(annotations_df["name"], as_index=False).agg(
        {
            "instrument": "first",
            "type": ", ".join,
            "rectified": list,
            "location": list,
            "values": list,
        }
    )
    return dcc.send_data_frame(
        df_final.to_csv, filename="annotation_record.csv", index=False
    )


@app.callback(
    Output("alert", "is_open"),
    Output("alert", "children"),
    Input("alert_state_all", "data"),
    Input("alert_state_stereo", "data"),
    Input("alert_state_color", "data"),
    prevent_initial_call=True,
)
def show_alert(all_alerts, stereo_alerts, color_alerts):
    """
    Displays an alert message based on the state data from different tabs.

    Args:
        all_alert_data: Alert state data from the 'all' tab.
        stereo_alert_data: Alert state data from the 'stereo' tab.
        color_alert_data: Alert state data from the 'color' tab.
    Returns:
        A tuple containing:
            - is_open (bool): Whether the alert should be open.
            - message (str): The alert message content.
    """
    if all_alerts is not None:
        return all_alerts["state"], all_alerts["msg"]
    elif stereo_alerts is not None:
        return stereo_alerts["state"], stereo_alerts["msg"]
    elif color_alerts is not None:
        return color_alerts["state"], color_alerts["msg"]
    else:
        raise PreventUpdate


if __name__ == "__main__":
    # To run with the development server (for debugging, etc.):
    # app.run(host="0.0.0.0", port=8050, debug=True) # Set debug=False for production-like testing with dev server
    logger.warning(
        "Starting server with Waitress on http://localhost:8050/ and http://127.0.0.1:8050/"
    )
    serve(app.server, host="0.0.0.0", port=8050)
