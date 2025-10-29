# C3 Data Review App - BlueOS Extension

A BlueOS extension for reviewing and annotating camera data from C3 surveys. This web-based tool supports stereo and color camera data analysis with calibration support.

For usage, see the official documentation at: [C3 Data Review App](https://github.com/vshie/BR-c3-data-review-app)

## BlueOS Extension Installation

This application is designed to run as a BlueOS extension. To install:

1. **Via BlueOS Extensions Manager** (when available in the Bazaar):
   - Open BlueOS web interface
   - Navigate to Extensions Manager
   - Search for "C3 Data Review App"
   - Click Install

2. **Manual Installation**:
   - Go to Extensions Manager → Installed tab
   - Click the "+" icon
   - Enter the following information:
     - **Extension Identifier**: `vshie.br-c3-data-review-app`
     - **Extension Name**: `C3 Data Review App`
     - **Docker image**: `vshie/br-c3-data-review-app`
     - **Docker tag**: `latest`
     - **Custom settings**: 
       ```json
       {
         "ExposedPorts": {
           "8050/tcp": {}
         },
         "HostConfig": {
           "Binds": [
             "/usr/blueos/extensions/c3-data-review:/app/data"
           ],
           "ExtraHosts": ["host.docker.internal:host-gateway"],
           "PortBindings": {
             "8050/tcp": [
               {
                 "HostPort": ""
               }
             ]
           },
           "NetworkMode": "host"
         }
       }
       ```

## Prerequisites
- BlueOS running on compatible hardware
- Camera data in the expected directory structure (left/, right/, center/ or Oak1Left/, Oak1Right/, Oak1Center/)

## Usage

1. **Data Preparation**: Place your camera data in the mounted data directory (`/usr/blueos/extensions/c3-data-review/` on the host)
2. **Access the App**: The extension will appear in the BlueOS sidebar once installed and running
3. **Load Data**: Use the dropdown to select your dataset folder
4. **Review Images**: Navigate through stereo, color, or all camera views
5. **Annotate**: Use the drawing tools to annotate images
6. **Export**: Download your annotations as CSV files

## Data Structure

The app expects camera data in one of these structures:
- `left/`, `right/`, `center/` (for standard stereo + color setup)
- `Oak1Left/`, `Oak1Right/`, `Oak1Center/` (for Oak camera naming convention)

Each directory should contain timestamped image files (`.jpg` format).

## Features

- **Multi-camera Support**: View stereo pairs and color images simultaneously
- **Calibration Support**: Upload and use camera calibration files
- **Image Rectification**: Toggle rectified/unrectified views
- **Annotation Tools**: Draw lines, rectangles, and shapes on images
- **Export Functionality**: Download annotations as CSV files
- **Responsive Interface**: Works on various screen sizes

## Development

### Building the Extension

The extension uses the official BlueOS deployment action for automated builds. To set up automated deployment:

1. **Configure GitHub Secrets** (in repository settings):
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: Your Docker Hub access token

2. **Configure GitHub Variables** (in repository settings):
   - `IMAGE_NAME`: `vshie/br-c3-data-review-app`
   - `MY_EMAIL`: `tonywhite@bluerobotics.com`
   - `MY_NAME`: `Tony White`
   - `ORG_EMAIL`: `support@bluerobotics.com`
   - `ORG_NAME`: `Blue Robotics`

3. **Manual Build** (for local testing):
   ```bash
   # Build the Docker image
   docker build -t vshie/br-c3-data-review-app .

   # Run locally for testing
   docker run -p 8050:8050 -v /path/to/your/data:/app/data vshie/br-c3-data-review-app
   ```

### Local Development

If running with Python directly (for development):

1. Clone the git directory
2. Navigate to the *app/* folder
3. Create a virtual python environment and install the required packages in *requirements.txt*
4. Run the python file *app.py*
5. Navigate to `http://localhost:8050/` to open the app