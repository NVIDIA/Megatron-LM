#! /bin/bash

set -x
env
eval "IMAGE=\$$IMAGE"

# Start a named container in detached mode
docker run -d --name download_test_data -w /workdir/ python:3.12-slim bash -c 'sleep infinity'
docker cp tests/. download_test_data:/workdir/tests
docker exec download_test_data bash -c '
    ls -al /workdir/
    pip install --no-cache-dir click requests
    python tests/test_utils/python_scripts/download_unit_tests_dataset.py --assets-dir ./assets
'
docker cp download_test_data:/workdir/assets ./
docker rm -f download_test_data

docker context create tls-environment
docker buildx create --name container --driver=docker-container --use tls-environment

ADDITIONAL_PARAMS=()

if [[ "$CI_COMMIT_BRANCH" == "ci-rebuild-mcore-nemo-image" || "$CI_COMMIT_BRANCH" == "main" || "$CI_COMMIT_BRANCH" == "dev" ]]; then
    ADDITIONAL_PARAMS+=("--pull")
fi

CI_COMMIT_BRANCH=$(echo "$CI_COMMIT_BRANCH" | tr '/' '-' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g')
ADDITIONAL_PARAMS+=("--cache-to type=registry,ref=${IMAGE}-buildcache:${CI_COMMIT_BRANCH}-${PLATFORM},mode=max")
ADDITIONAL_PARAMS+=("--cache-from type=registry,ref=${IMAGE}-buildcache:${CI_COMMIT_BRANCH}-${PLATFORM}")
ADDITIONAL_PARAMS+=("-t ${IMAGE}:${CI_COMMIT_BRANCH}-${PLATFORM}")

if [[ -n "$CI_MERGE_REQUEST_IID" ]]; then
    ADDITIONAL_PARAMS+=("--cache-to type=registry,ref=${IMAGE}-buildcache:${CI_MERGE_REQUEST_IID}-${PLATFORM},mode=max")
    ADDITIONAL_PARAMS+=("--cache-from type=registry,ref=${IMAGE}-buildcache:${CI_MERGE_REQUEST_IID}-${PLATFORM}")
    ADDITIONAL_PARAMS+=("-t ${IMAGE}:${CI_MERGE_REQUEST_IID}-${PLATFORM}")
fi

if [[ "$CI_COMMIT_BRANCH" == "ci-nightly" ]]; then
    ADDITIONAL_PARAMS+=("-t ${IMAGE}:nightly-${PLATFORM}")
fi

if [[ -n "$TE_GIT_REF" ]]; then
    ADDITIONAL_PARAMS+=("--build-arg TE_COMMIT=${TE_GIT_REF}")
fi

echo $(git rev-parse HEAD)

JET_API_VERSION=$(curl -s -u "$ARTIFACTORY_USER:$ARTIFACTORY_TOKEN" "https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hw-joc-pypi/simple/jet-api/" | grep -o 'href="../../jet-api/[0-9.]*/' | sed 's|href="../../jet-api/||;s|/||' | sort -V -r | head -n1)

DOCKER_BUILDKIT=1 docker build \
    --secret id=JET_INDEX_URLS \
    --secret id=LOGGER_INDEX_URL \
    --target $STAGE \
    -f docker/$FILE \
    -t ${IMAGE}:${CI_PIPELINE_ID}-${PLATFORM} \
    --builder=container \
    --build-arg JET_API_VERSION=$JET_API_VERSION \
    --build-arg FROM_IMAGE_NAME=$BASE_IMAGE \
    --provenance=false \
    --push \
    --progress plain \
    ${ADDITIONAL_PARAMS[@]} .
