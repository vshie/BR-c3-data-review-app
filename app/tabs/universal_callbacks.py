from dash import Dash, html, dcc, callback, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import dash_extensions as de
from typing import List, Optional, Dict


class UniversalCallbacks:
    """
    Manages universal callbacks that are common across multiple tabs,
    such as next/previous image navigation.
    """

    def __init__(self, suffixes: List[str], app: Dash):
        """
        Initializes the UniversalCallbacks.

        Args:
            suffixes: A list of string suffixes used for component IDs in different tabs.
            app: The main Dash application instance.
        """
        self.suffixes: List[str] = suffixes
        self.app = app

    def create_callbacks(self) -> None:
        """
        Defines and registers universal callbacks for each specified suffix.
        This typically includes callbacks for navigating between images (next/previous).
        """
        for suffix in self.suffixes:

            @self.app.callback(
                Output("index_{}".format(suffix), "data"),
                State("data_paths", "data"),
                Input("next_{}".format(suffix), "n_clicks"),
                Input("next_key_{}".format(suffix), "n_keydowns"),
                Input("previous_{}".format(suffix), "n_clicks"),
                Input("previous_key_{}".format(suffix), "n_keydowns"),
            )
            def update_index(data, n_clicks, n_keys, p_clicks, p_keys):
                """
                Updates the current image index based on next/previous button clicks
                or key presses.

                Args:
                    data: A list of dictionaries containing data paths, typically from dcc.Store.
                          Each dictionary should represent a row of data.
                    n_clicks: Number of times the 'next' button was clicked.
                    n_keys: Number of times the 'next' key (e.g., ArrowRight) was pressed.
                    p_clicks: Number of times the 'previous' button was clicked.
                    p_keys: Number of times the 'previous' key (e.g., ArrowLeft) was pressed.

                Returns:
                    The updated image index.

                Raises:
                    PreventUpdate: If no input has triggered the callback or if data is None.
                """
                if data is None:
                    raise PreventUpdate

                path_data = pd.DataFrame(data)
                if (
                    n_clicks is None
                    and n_keys is None
                    and p_clicks is None
                    and p_keys is None
                ):
                    raise PreventUpdate
                else:
                    index = (
                        int(n_clicks or 0)
                        + int(n_keys or 0)
                        - int(p_clicks or 0)
                        - int(p_keys or 0)
                    )
                    if index < 0:
                        index = 0
                    if index >= len(path_data):
                        index = len(path_data) - 1
                return index
