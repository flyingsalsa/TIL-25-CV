# syntax=docker/dockerfile:1

# NOTE: Docker's GPU support works for most images despite common misconceptions.
FROM python:3.12-slim-bookworm AS deploy

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

# Cache packages to speed up builds, see: https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/reference.md#run---mounttypecache
# Example:
# RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
#   echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
# RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
#   --mount=type=cache,target=/var/lib/apt,sharing=locked \
#   apt-get update && apt-get install -y --no-install-recommends \
#   python3-pip
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
  pip install -U pip

WORKDIR /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Remember to regenerate requirements.txt!
COPY --link requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
  pip install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
# Copy files as needed in order of largest/least changed to smallest/most changed.
COPY --link models ./models
COPY --link src ./src

EXPOSE 5002
# uvicorn --host=0.0.0.0 --port=5002 --factory src:create_app
CMD ["uvicorn", "--host=0.0.0.0", "--port=5002", "--factory", "src:create_app"]
