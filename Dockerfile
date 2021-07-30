# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7.11-buster

# Copy requirements.txt to the docker image and install packages.
COPY requirements.txt /
RUN pip install -r requirements.txt

# Install production dependencies.

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN python ./OpenNMT-py/setup.py install

# Expose Port 5000
EXPOSE 5000
ENV PORT 5000


# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD cd OpenNMT-py && exec gunicorn --bind :$PORT --workers 1 --threads 1 --timeout 0 wsgi:app
