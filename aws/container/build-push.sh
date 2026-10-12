#!/bin/bash
# Construye la imagen del modelo para SageMaker (linux/amd64) y la publica en
# el repositorio ECR del proyecto, con el tag de container/image-version.js.
# Si ese tag ya existe no construye nada (los tags son inmutables).
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
SSM_PREFIX=/mlops-demand-forecasting-image

TAG=$(node -e "console.log(require('$PROJECT_DIR/container/image-version.js').imageVersion())")
REPOSITORY_URI=$(aws ssm get-parameter --name "$SSM_PREFIX/RepositoryUri" --query Parameter.Value --output text)
REPOSITORY_NAME=${REPOSITORY_URI#*/}
REGISTRY=${REPOSITORY_URI%%/*}
IMAGE=$REPOSITORY_URI:$TAG

if aws ecr describe-images --repository-name "$REPOSITORY_NAME" --image-ids "imageTag=$TAG" > /dev/null 2>&1; then
    echo "Image $IMAGE already pushed. Nothing to do."
    exit 0
fi

aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"

# El hosting de SageMaker (CreateModel) solo acepta manifiestos Docker v2, y un
# `docker push` desde el image store containerd de Docker Desktop sube OCI. Un
# builder buildx dedicado exporta media types Docker explícitos, sin
# atestaciones de provenance/SBOM, para las instancias x86 de SageMaker.
BUILDER=mlops-builder
if ! docker buildx inspect "$BUILDER" > /dev/null 2>&1; then
    docker buildx create --name "$BUILDER" --driver docker-container > /dev/null
fi
docker buildx build --builder "$BUILDER" --platform linux/amd64 \
    --provenance=false --sbom=false \
    --output "type=image,name=$IMAGE,push=true,oci-mediatypes=false" \
    -f "$PROJECT_DIR/container/Dockerfile" "$PROJECT_DIR"

MEDIA_TYPE=$(aws ecr describe-images --repository-name "$REPOSITORY_NAME" --image-ids "imageTag=$TAG" \
    --query "imageDetails[0].imageManifestMediaType" --output text)
if [ "$MEDIA_TYPE" != "application/vnd.docker.distribution.manifest.v2+json" ]; then
    echo "ERROR: $IMAGE was pushed with manifest $MEDIA_TYPE; SageMaker endpoints need Docker v2." >&2
    exit 1
fi
echo "Pushed $IMAGE ($MEDIA_TYPE)"
