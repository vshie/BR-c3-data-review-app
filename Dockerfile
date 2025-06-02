FROM python:3.10

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir /app

WORKDIR /app

COPY app/assets assets
COPY app/tabs tabs
COPY app/utils utils
COPY app/app.py .
COPY app/requirements.txt .

RUN pip3 install -r requirements.txt

EXPOSE 8050

CMD ["python3", "app.py"]