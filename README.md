# C3 Data Review App

For usage, see the official documentation at: [C3 Data Review App](https://c3-data-review-app-157680.gitlab.io/)


## Prerequisites
If running with Docker, this system requires Docker to be installed
If running with Python, this system requires Python 3+.

## Installation

There are two ways to run this application:
1. Building and running a docker image
2. Running the python *app.py* script locally

### To run with Docker: 
1. Clone the git directory
2. Navigate to the directory
3. Build the docker image:  
`docker build -t c3-data-review-app .`
4. Run the docker image  

You'll need to change the volume mounts to match your filesystem and platform. It is recommended that you mount the folder that contains all of your datasets. For example, the folder */data/c3-app* might contain multiple dataset folders: */data/c3-app/c3-survey-1* and */data/c3-app/c3-survey-2*. Note: If you mount a folder with too many subdirectories, it may take a while for the app to load.

  | Platform           | Command Example                                                     |
  | ------------------ | ------------------------------------------------------------------- |
  | Linux/macOS        | `docker run -p 8050:8050 -v /data/c3-app:/mnt/data c3-data-review-app`    |
  | Windows CMD        | `docker run -p 8050:8050 -v C:\data\c3-app:/mnt/data c3-data-review-app`  |
  | Windows PowerShell | `docker run -p 8050:8050 -v //c/data/c3-app:/mnt/data c3-data-review-app` |
    
  The available folders from the mounted directory will be listed in dropdown for data selection. If the desired folder is not shown, please revisit the directory mounting instructions.

5. Navigate to `http://localhost:8050/ or http://127.0.0.1:8050/` to open the app.

### To run locally:
1. Clone the git directory
2. Navigate to the directory
3. Navigate to the *app/* folder
4. Create a virtual python environment and install the required packages in *requirements.txt*, or install the packages with your global python distribution (not recommended).
5. Run the python file *app.py*
6. Navigate to `http://localhost:8050/ or http://127.0.0.1:8050/` to open the app.