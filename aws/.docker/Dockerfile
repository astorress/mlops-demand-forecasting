######## BASE IMAGE ########
# Imagen del devcontainer del proyecto AWS (target img-demandforecasting, ver
# .docker/docker-compose.yml).
FROM python:3.13-slim-bookworm AS python-base
FROM node:24-bookworm-slim AS img-base

ARG USER=mlops
ARG USER_HOME=/home/$USER
ARG VOLUME=/app
ARG SHELL=/bin/bash

COPY --from=python-base /usr/local/ /usr/local/
RUN ldconfig
RUN pip3 install --upgrade pip

RUN apt-get update
RUN npm install -g npm

# Configuración inicial del sistema operativo
RUN apt-get install -y locales \
    && sed -i 's/^# *\(en_US.UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen
RUN apt-get install -y sudo lsb-release git curl zip unzip jq

# Dependencias de las extensiones de VS Code (validación de CloudFormation)
RUN pip3 install cfn-lint cfn-lint-serverless

# AWS CLI v2
RUN case "$(dpkg --print-architecture)" in \
        "amd64") AWSCLI_ARCH="x86_64" ;; \
        "arm64"|"aarch64") AWSCLI_ARCH="aarch64" ;; \
        *) echo "Unsupported architecture" && exit 1 ;; \
    esac; \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-${AWSCLI_ARCH}.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws \
    && apt-get install -y less

# Serverless Framework v4 (los plugins se instalan por subproyecto, nunca global)
RUN npm install -g serverless

# Usuario no root
RUN echo "$USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN useradd -ms $SHELL $USER
RUN mkdir -p $VOLUME \
    && chown -R $USER:$USER $VOLUME

# Historial de bash persistente (volumen /commandhistory)
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && echo $SNIPPET >> "/root/.bashrc" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USER /commandhistory \
    && echo $SNIPPET >> "$USER_HOME/.bashrc"

# El usuario puede actualizar paquetes npm globales (serverless) en runtime
RUN chown -R $USER:$USER /usr/local/lib/node_modules /usr/local/bin


######## ML BASE IMAGE ########
# Herramientas comunes de los proyectos de ML: tests, linters, notebooks, SDK
# de AWS y el CLI de Docker (construye/publica la imagen del modelo con el
# daemon del host, montado en /var/run/docker.sock). Las librerías de ML NO
# se instalan aquí: cada proyecto fija las suyas en container/requirements.txt,
# compartido con su imagen de SageMaker.
FROM img-base AS img-ml-base

# Docker CLI + buildx (repositorio oficial de Docker para Debian)
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y docker-ce-cli docker-buildx-plugin

# libsqlite3: pytest-cov / coverage · libgomp1: OpenMP de LightGBM
RUN apt-get install -y libsqlite3-0 libgomp1

RUN pip3 install pylint yapf isort
RUN pip3 install boto3 jupyterlab
RUN pip3 install pytest pytest-cov pytest-env pytest-mock moto


######## DEMAND FORECASTING IMAGE ########
# Contexto de build: aws/ (ver .docker/docker-compose.yml).
FROM img-ml-base AS img-demandforecasting

# Mismo runtime de ML (versiones fijas) que la imagen del modelo
COPY container/requirements.txt /tmp/demand-forecasting-requirements.txt
RUN pip3 install -r /tmp/demand-forecasting-requirements.txt

USER $USER
