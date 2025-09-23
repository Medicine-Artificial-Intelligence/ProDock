# ----------------------------
# Stage 1: build wheel (pip available here)
# ----------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m pip install --upgrade pip setuptools wheel build hatchling

COPY pyproject.toml README.md CHANGELOG.md ./
COPY prodock/ ./prodock
COPY requirements.txt ./requirements.txt

RUN python -m build --wheel --no-isolation

# ----------------------------
# Stage 2: runtime (micromamba + conda env)
# ----------------------------
FROM mambaorg/micromamba:1.4.0 AS runtime

ARG CONDA_ENV=prodock
ARG PYTHON_VERSION=3.11
ARG OPENMM_PACKAGE="openmm=8.3.1"
ARG PDBFIXER_PACKAGE="pdbfixer"
ARG CONDA_CHANNEL=conda-forge

WORKDIR /opt/prodock

# Copy artifacts from builder
COPY --from=builder /build/dist/*.whl ./
COPY --from=builder /build/requirements.txt ./requirements.txt

# ---- RUN APT AS ROOT (needed for permissions) ----
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential git pkg-config \
      libblas-dev liblapack-dev gfortran ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# Switch back to the default micromamba user
USER $MAMBA_USER

# Create conda env and install heavy binary deps from conda-forge
RUN micromamba create -n ${CONDA_ENV} -y -c ${CONDA_CHANNEL} \
      python=${PYTHON_VERSION} \
      rdkit openbabel pymol-open-source vina py3Dmol mdanalysis \
      ${OPENMM_PACKAGE} ${PDBFIXER_PACKAGE} \
      libstdcxx-ng libgcc-ng \
  && micromamba clean --all --yes

# Make env the default PATH
ENV CONDA_BASE=/opt/conda
ENV CONDA_ENV_PREFIX=${CONDA_BASE}/envs/${CONDA_ENV}
ENV PATH=${CONDA_ENV_PREFIX}/bin:${PATH}

# Filter out conda-installed packages so pip doesn't try to (re)build them
RUN set -eux; \
  grep -v -E '^(rdkit|openbabel-wheel|openbabel|pymol-open-source-whl|pymol-open-source|vina|py3Dmol|openmm|pdbfixer|mdanalysis)' requirements.txt > reqs-pip.txt || true; \
  echo "== pip reqs to install (filtered) =="; cat reqs-pip.txt || true

# Install pip-only deps inside env, then install your wheel without resolving heavy deps
RUN set -eux; \
  micromamba run -n ${CONDA_ENV} python -m pip install --upgrade pip setuptools wheel; \
  if [ -s reqs-pip.txt ]; then \
    micromamba run -n ${CONDA_ENV} pip install --no-cache-dir -r reqs-pip.txt; \
  else \
    echo "No pip requirements after filtering."; \
  fi; \
  micromamba run -n ${CONDA_ENV} pip install --no-cache-dir --no-deps ./*.whl; \
  rm -f ./*.whl

CMD ["micromamba","run","-n","prodock","python","-c","import importlib.metadata as m; print('prodock==', m.version('prodock'))"]
