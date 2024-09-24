FROM cnstark/pytorch:2.0.1-py3.10.11-cuda11.8.0-ubuntu22.04 

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm/input /opt/algorithm/output /opt/algorithm/test \
    && chown algorithm:algorithm /opt/algorithm/input /opt/algorithm/output /opt/algorithm/test
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm ultralytics/ /opt/algorithm/ultralytics/
COPY --chown=algorithm:algorithm best.pt /opt/algorithm/
# COPY --chown=algorithm:algorithm test/ /opt/algorithm/test/
# COPY --chown=algorithm:algorithm output/results.json /opt/algorithm/output/results.json


ENTRYPOINT python -m process $0 $@
