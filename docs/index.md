# C3 Data Review App

MarineSitu C3 camera data review and annotation tool.

## Introduction

This application provides a user-friendly interface to streamline the process of inspecting footage and sensor data from [MarineSitu C3 devices](https://www.marinesitu.com/pagec3-1). Whether you're working on computer vision projects, robotics, or any field that utilizes the rich data from MarineSitu cameras, the C3 Data Review App aims to simplify your data interaction workflow.

Key goals of C3 Data Review App include:  

-   **Efficient Data Review:** Quickly navigate through recorded frames from one, two, or all three cameras, and visualize associated metadata.  
-   **Precise Annotation:** Tools to accurately measure and label objects, events, or regions of interest within your MarineSitu datasets.  
-   **Streamlined Workflow:** An intuitive experience from loading data to exporting your annotations.  

This project uses [OpenCV](https://opencv.org/) for rectification and stereo calculations in conjunction with a luxonis-style device calibration file (see example file in the repository for formatting).

This documentation will guide you through the installation and usage of the C3 Data Review App.

## Installation

Currently, this runs from building a Dockerfile. 

1. Clone the git directory
2. Navigate to the directory
3. Build the docker image:  
`docker build -t c3-data-review-app .`
4. Run the docker image  
You'll need to change the volume mounts to match your filesystem and platform

    | Platform           | Command Example                                                     |
    | ------------------ | ------------------------------------------------------------------- |
    | Linux/macOS        | `docker run -p 8050:8050 -v /data/c3-app:/mnt/data c3-data-review-app`    |
    | Windows CMD        | `docker run -p 8050:8050 -v C:\data\c3-app:/mnt/data c3-data-review-app`  |
    | Windows PowerShell | `docker run -p 8050:8050 -v //c/data/c3-app:/mnt/data c3-data-review-app` |
    
    Ensure that filepaths entered during use of the app are relative to this user specified volume:  
    
        `"C:/data/c3-app/c3-survey-data/"` becomes `"c3-survey-data"`

5. Navigate to `http://localhost:8050/ or http://127.0.0.1:8050/` to open the app.
