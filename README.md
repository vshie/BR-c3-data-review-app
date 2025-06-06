# C3 Data Review App

For usage, see the official documentation at: [C3 Data Review App](https://c3-data-review-app-157680.gitlab.io/)


## Prerequisites
This system requires docker to be installed

## Installation

Currently, this runs from building a Dockerfile. 

1. Clone the git directory
2. Navigate to the directory
3. Build the docker image:  
`docker build -t c3-data-review-app .`
4. Run the docker image  
You'll need to change the volume mount to match your filesystem and platform, specifically the section: `/data/oak-app`  

    | Platform           | Command Example                                                     |
    | ------------------ | ------------------------------------------------------------------- |
    | Linux/macOS        | `docker run -p 8050:8050 -v /data/c3-data:/mnt/data c3-data-review-app`    |
    | Windows CMD        | `docker run -p 8050:8050 -v C:\data\c3-data:/mnt/data c3-data-review-app`  |
    | Windows PowerShell | `docker run -p 8050:8050 -v //c/data/c3-data:/mnt/data c3-data-review-app` |
    
    Ensure that filepaths entered during use of the app are relative to this user specified volume:  
    
    `"C:/data/oak-app/oak-survey-data/"` becomes `"oak-survey-data"`

5. Navigate to `http://localhost:8050/` or `http://127.0.0.1:8050/` to open the app.