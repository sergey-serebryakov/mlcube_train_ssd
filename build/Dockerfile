FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Set working directory
WORKDIR /mlperf/ssd

# Necessary zone info for tzdata
ENV TZ=America/New_York
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3-tk python-pip numactl git curl wget numactl

RUN pip install --upgrade pip

# Necessary pip packages
COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir cython \
 && pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip \
 && pip install --no-cache-dir -r /requirements.txt

# Copy SSD code
COPY ssd .
COPY main.py .
RUN chmod +x ./run_and_time.sh

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ENTRYPOINT ["python", "/mlperf/ssd/main.py"]
