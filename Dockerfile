# Multi-stage Dockerfile using micromamba (recommended for heavy scientific packages)
# Stage: build wheel
FROM mambaorg/micromamba:1.4.0 AS builder

ARG CONDA_CHANNEL=conda-forge
WORKDIR /build

# Install pip tooling for building wheel
RUN pip install --upgrade pip build hatchling

# Copy project metadata + source
COPY pyproject.toml README.md CHANGELOG.md ./
COPY prodock/ ./prodock
# copy requirements so we can filter later
COPY requirements.txt ./

# Build a wheel (in /build/dist)
RUN python -m build --wheel --no-isolation

#########################################
# Stage: runtime image with conda env
#########################################
FROM mambaorg/micromamba:1.4.0

ARG CONDA_ENV=prodock
ARG PYTHON_VERSION=3.11

WORKDIR /opt/prodock

# Copy built wheel(s) and requirements from builder
COPY --from=builder /build/dist/*.whl ./
COPY --from=builder /build/requirements.txt ./requirements.txt

# Create environment and install heavy binary packages from conda-forge
# Adjust the package list below to match what you need from conda-forge.
# We include rdkit, openbabel, pymol-open-source, vina, py3Dmol where available.
RUN micromamba create -n ${CONDA_ENV} -y -c conda-forge \
      python=${PYTHON_VERSION} \
      rdkit openbabel pymol-open-source vina py3Dmol libstdcxx-ng libgcc-ng \
  && micromamba clean --all --yes

# Export the env prefix for convenience
ENV MAMBA_EXE=/usr/bin/micromamba
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_BASE=/opt/conda
ENV CONDA_ENV_PREFIX=${CONDA_BASE}/envs/${CONDA_ENV}
ENV PATH=${CONDA_ENV_PREFIX}/bin:$PATH

# Filter requirements.txt to remove packages we installed with conda.
# Adjust the pattern if you change the conda-installed package list.
RUN set -eux; \
    # create filtered pip reqs (exclude heavy conda-installed names)
    grep -v -E '^(rdkit|openbabel-wheel|openbabel|pymol-open-source-whl|pymol-open-source|vina|py3Dmol)' requirements.txt > reqs-pip.txt || true; \
    echo "== pip reqs to install (filtered) =="; cat reqs-pip.txt || true

# Install pip-only (or non-conda) requirements inside the conda env, then install your wheel
# We use micromamba run -n ... pip to ensure pip runs inside the created env.
RUN set -eux; \
    micromamba run -n ${CONDA_ENV} pip install --upgrade pip setuptools wheel; \
    if [ -s reqs-pip.txt ]; then \
      micromamba run -n ${CONDA_ENV} pip install --no-cache-dir -r reqs-pip.txt; \
    else \
      echo "No pip requirements after filtering."; \
    fi; \
    # install the package wheel WITHOUT letting pip try to re-resolve heavy deps (they are in conda).
    micromamba run -n ${CONDA_ENV} pip install --no-cache-dir --no-deps ./*.whl; \
    rm -f ./*.whl

# Sanity check: print prodock version when container is run
CMD ["micromamba","run","-n","prodock","python","-c","import importlib.metadata as m; print('prodock==', m.version('prodock'))"]
