#! /bin/bash

set -x
env
eval "IMAGE=\$$IMAGE"

docker context create tls-environment
docker buildx create --name container --driver=docker-container --use tls-environment

ADDITIONAL_PARAMS=()

if [[ "$CI_COMMIT_BRANCH" == "ci-rebuild-mcore-nemo-image" || "$CI_COMMIT_BRANCH" == "main" ]]; then
    ADDITIONAL_PARAMS+=("--pull")
    ADDITIONAL_PARAMS+=("--cache-to type=registry,ref=${IMAGE}-buildcache:main,mode=max")
else
    ADDITIONAL_PARAMS+=("--cache-to type=registry,ref=${IMAGE}-buildcache:${CI_MERGE_REQUEST_IID:-$CI_COMMIT_REF_SLUG},mode=max")
fi

if [[ "$CI_COMMIT_BRANCH" == "ci-nightly" ]]; then
    ADDITIONAL_PARAMS+=("-t ${IMAGE}:nightly")
fi

echo $(git rev-parse HEAD)

JET_API_VERSION=$(curl -s -u "$ARTIFACTORY_USER:$ARTIFACTORY_TOKEN" "https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hw-joc-pypi/simple/jet-api/" | grep -o 'href="../../jet-api/[0-9.]*/' | sed 's|href="../../jet-api/||;s|/||' | sort -V -r | head -n1)

DOCKER_BUILDKIT=1 docker build \
    --secret id=JET_INDEX_URLS \
    --secret id=LOGGER_INDEX_URL \
    --secret id=EXPERIMENTAL_FLASH_ATTN \
    --target $STAGE \
    -f $FILE \
    -t ${IMAGE}:${CI_PIPELINE_ID} \
    --builder=container \
    --build-arg JET_API_VERSION=$JET_API_VERSION \
    --cache-from type=registry,ref=${IMAGE}-buildcache:${CI_MERGE_REQUEST_IID} \
    --cache-from type=registry,ref=${IMAGE}-buildcache:main \
    --build-arg FROM_IMAGE_NAME=$BASE_IMAGE \
    --push \
    --progress plain \
    ${ADDITIONAL_PARAMS[@]} .
