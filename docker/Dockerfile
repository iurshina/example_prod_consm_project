FROM python:3.10

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Set the working directory to /app
WORKDIR /app

COPY laser_app/requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# Download LASER from FB
RUN git clone https://github.com/facebookresearch/LASER.git

ENV LASER /app/LASER
WORKDIR $LASER

RUN bash ./install_models.sh

RUN bash ./install_external_tools.sh

WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:${LASER}/source/"

EXPOSE 80

COPY laser_app/server.py /app/server.py

CMD [ "python", "server.py" ]