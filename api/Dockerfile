# Use Python 3.8.13 as the base image
FROM python:3.8.13 as base

# Arguments for user and group IDs
ARG UID
ARG GID

# Add a non-root user
RUN addgroup --gid $GID app \
    && adduser --disabled-login --geco '' --uid $UID --gid $GID app

# Switch to the non-root user
USER app

# Append /home/app/.local/bin/ to PATH variable because
# gunicorn is installed there.
ENV PYTHONPATH=$PYTHONPATH:/src/
ENV PATH=$PATH:/home/app/.local/bin/

# Set the working directory
WORKDIR /src

# Copy the application source code
COPY ./ /src

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Build stage
FROM base as build

# Set the entry point for the application
ENTRYPOINT ["gunicorn", "--workers=8", "--bind", "0.0.0.0:5000", "--timeout", "300", "app:app"]
