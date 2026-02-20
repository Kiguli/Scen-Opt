# Docker Deployment

The easiest way to run Scen-O-Con locally is with Docker. No Python installation or dependency management required.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your machine

## Quick Start

Build the image:

```bash
docker build -t scen-o-con .
```

Run the container:

```bash
docker run -p 5000:5000 scen-o-con
```

Open your browser at **http://localhost:5000**.

## Using a MOSEK License

MOSEK is a commercial solver included in the dependencies. It works without a license for the other solvers, but if you want to use MOSEK specifically, mount your license file:

```bash
docker run -p 5000:5000 \
  -v /path/to/mosek.lic:/home/appuser/mosek/mosek.lic:ro \
  scen-o-con
```

Alternatively, you can upload your MOSEK license through the web UI when solving a problem.

## Configuration

### Custom Port

To run on a different port (e.g. 8080):

```bash
docker run -p 8080:5000 scen-o-con
```

Then open http://localhost:8080.

### More Workers

For handling more concurrent requests:

```bash
docker run -p 5000:5000 -e GUNICORN_CMD_ARGS="--workers 4" scen-o-con
```

## Rebuilding

After pulling new changes or modifying code, rebuild the image:

```bash
docker build -t scen-o-con .
```
