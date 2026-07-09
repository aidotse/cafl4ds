# syntax=docker/dockerfile:1.4
#
# A Dockerfile that encapsulates the entire operative codebase as a standalone environment
#
# Example run with no private dependencies:
# docker build --secret id=internal_ca,src=$CERTS --build-arg VERSION=... -f docker/Dockerfile -t cafl4ds:latest .
#
# If you need private dependencies, you need to authenticate uv via build secrets. Example:
# docker build --secret id=internal_ca,src=$CERTS --secret id=uv_username,env=$UV_USERNAME --secret id=uv_password,env=$UV_PASSWORD --build-arg VERSION=... -f docker/Dockerfile -t cafl4ds:latest .
#
# CERTS:            <your/PEM/or/CRT/file>. Adding this --secret is optional.
# UV_USERNAME:      Gitlab username
# UV_PASSWORD:      PAT (for this repo)

# ---- STAGE 1: Base ------------------------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS base

# Certificates
RUN --mount=type=secret,id=internal_ca \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    if [ -f /run/secrets/internal_ca ]; then \
        cp /run/secrets/internal_ca /usr/local/share/ca-certificates/internal-ca.crt && \
        update-ca-certificates; \
    fi && \
    rm -rf /var/lib/apt/lists/*

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    UV_CERT_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# ---- STAGE 2: Builder ------------------------------------------------------------------------------------------------
FROM base AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /bin/

# Optimizations for uv in build stage
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# Hatchling requires a version number to package / install with the code. Pass via --build-arg
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION

WORKDIR /app

# Because of --no-install-project, this dependency layer will only invalidate when we install new dependencies, not
# change our source code
COPY pyproject.toml uv.lock ./

# IF YOU NEED PRIVATE PACKAGES:
# 1. Uncomment the --mount and UV_INDEX lines below.
# 2. Add the corresponding --secret flags to your docker build command in CI.
RUN --mount=type=cache,target=/root/.cache/uv \
    #--mount=type=secret,id=uv_username \
    #--mount=type=secret,id=uv_password \
    #UV_INDEX_CAFL4DS_USERNAME=$(cat /run/secrets/uv_username) \
    #UV_INDEX_CAFL4DS_PASSWORD=$(cat /run/secrets/uv_password) \
    uv sync --frozen --no-dev --no-install-project

# Application layer. `--no-editable`: installs our project dependency straight into `site-packages` = no messing around
# with env variables to find the package
# SETUPTOOLS_SCM_PRETEND_VERSION tells Hatchling to pretend this is the version instead of looking for .git folder
COPY README.md LICENSE.md ./
COPY ./cafl4ds ./cafl4ds

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


# ---- STAGE 3: Production ---------------------------------------------------------------------------------------------
FROM base AS production

WORKDIR /app

COPY --link --from=builder /app/.venv /app/.venv
COPY --link --from=builder /app/cafl4ds /app/cafl4ds

# Run as a non-root user. --chown is awkward to combine reliably with --link across
# BuildKit versions, so fix ownership with a plain chown after the copy instead.
RUN groupadd --system app && useradd --system --gid app --no-create-home app && \
    chown -R app:app /app
USER app

ENV PATH="/app/.venv/bin:$PATH"
