FROM python:3.10

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy app files
COPY app/assets assets
COPY app/tabs tabs
COPY app/utils utils
COPY app/app.py .
COPY app/requirements.txt .
COPY app/register_service .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Create directory for data storage
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8050

LABEL version="1.0.0"

ARG IMAGE_NAME
LABEL permissions='\
{\
  "ExposedPorts": {\
    "8050/tcp": {}\
  },\
  "HostConfig": {\
    "Binds": [\
      "/usr/blueos/extensions/c3-data-review:/app/data"\
    ],\
    "ExtraHosts": ["host.docker.internal:host-gateway"],\
    "PortBindings": {\
      "8050/tcp": [\
        {\
          "HostPort": ""\
        }\
      ]\
    },\
    "NetworkMode": "host"\
  }\
}'

ARG AUTHOR
ARG AUTHOR_EMAIL
LABEL authors='[\
    {\
        "name": "Tony White",\
        "email": "tonywhite@bluerobotics.com"\
    }\
]'

ARG MAINTAINER
ARG MAINTAINER_EMAIL
LABEL company='\
{\
        "about": "C3 Data Review App for analyzing camera data",\
        "name": "Blue Robotics",\
        "email": "support@bluerobotics.com"\
    }'
LABEL type="tool"

ARG REPO
ARG OWNER
LABEL readme='C3 Data Review App - A web-based tool for reviewing and annotating camera data from C3 surveys. Supports stereo and color camera data analysis with calibration support.'
LABEL links='\
{\
        "source": "https://github.com/vshie/BR-c3-data-review-app"\
    }'
LABEL requirements="core >= 1.1"

ENTRYPOINT ["python3", "-u", "/app/app.py"]