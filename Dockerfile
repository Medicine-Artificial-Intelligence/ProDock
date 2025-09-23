# ----------------------------
# Multi-stage Dockerfile (preferred: conda binary installs + small fallback)
# ----------------------------
# Stage 1: build wheel (use stock python image so pip is available)
FROM python:3.11-slim AS builder

# ensure basic build tools (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# install python build tooling
RUN python -m pip install --upgrade pip setuptools wheel build hatchling

# copy project metadata and source
COPY pyproject.toml README.md CHANGELOG.md ./
COPY prodock/ ./prodock
COPY requirements.txt ./requirements.txt

# build the wheel into /build/dist
RUN python -m build --wheel --no-isolation

# ----------------------------
# Stage 2: runtime image with micromamba / conda env
# ----------------------------
FROM mambaorg/micromamba:1.4.0 AS runtime

ARG CONDA_ENV=prodock
ARG PYTHON_VERSION=3.11
ARG OPENMM_PACKAGE="openmm=8.3.1"
ARG PDBFIXER_PACKAGE="pdbfixer"
ARG CONDA_CHANNEL=conda-forge

WORKDIR /opt/prodock

# copy wheel and requirements from builder
COPY --from=builder /build/dist/*.whl ./
COPY --from=builder /build/requirements.txt ./requirements.txt

# Install minimal system build tools as a fallback (so pip can compile if absolutely necessary)
# This helps avoid "gcc: command not found" when pip must compile something.
RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential git pkg-config \
     libblas-dev liblapack-dev gfortran ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Create conda env and install heavy binary packages from conda-forge,
# including MDAnalysis, OpenMM and PDBFixer (now included).
# Adding mdanalysis here avoids pip building it from source.
RUN micromamba create -n ${CONDA_ENV} -y -c ${CONDA_CHANNEL} \
      python=${PYTHON_VERSION} \
      rdkit openbabel pymol-open-source vina py3Dmol mdanalysis \
      ${OPENMM_PACKAGE} ${PDBFIXER_PACKAGE} \
      libstdcxx-ng libgcc-ng \
  && micromamba clean --all --yes

# expose conda env prefix and PATH for runtime
ENV CONDA_BASE=/opt/conda
ENV CONDA_ENV_PREFIX=${CONDA_BASE}/envs/${CONDA_ENV}
ENV PATH=${CONDA_ENV_PREFIX}/bin:${PATH}

# Filter requirements.txt to remove packages installed via conda
# Update the grep pattern if you change what conda installs.
RUN set -eux; \
    grep -v -E '^(rdkit|openbabel-wheel|openbabel|pymol-open-source-whl|pymol-open-source|vina|py3Dmol|openmm|pdbfixer|mdanalysis)' requirements.txt > reqs-pip.txt || true; \
    echo "== pip reqs to install (filtered) =="; cat reqs-pip.txt || true

# Install pip-only requirements inside the conda env, then install the built wheel (no-deps)
RUN set -eux; \
    micromamba run -n ${CONDA_ENV} python -m pip install --upgrade pip setuptools wheel; \
    if [ -s reqs-pip.txt ]; then \
      micromamba run -n ${CONDA_ENV} pip install --no-cache-dir -r reqs-pip.txt; \
    else \
      echo "No pip requirements after filtering."; \
    fi; \
    # install wheel without letting pip re-resolve heavy deps (they're already in conda env)
    micromamba run -n ${CONDA_ENV} pip install --no-cache-dir --no-deps ./*.whl; \
    rm -f ./*.whl

# Default runtime check
CMD ["micromamba","run","-n","prodock","python","-c","import importlib.metadata as m; print('prodock==', m.version('prodock'))"]
