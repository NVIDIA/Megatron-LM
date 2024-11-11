import org.apache.commons.io.FilenameUtils
import groovy.json.JsonOutput


def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME" || true
        lsb_release -sd || true
        uname -r || true
        cat /sys/module/amdgpu/version || true
        ls /opt/ -la || true
    """
}

def clean_up_docker() {
    sh 'docker ps -a || true' // "|| true" suppresses errors
    sh 'docker kill $(docker ps -q) || true'
    sh 'docker rm $(docker ps -a -q) || true'
    sh 'docker rmi $(docker images -q) || true'
    sh 'docker system prune -af --volumes || true'
}

def clean_docker_build_cache() {
    sh 'docker system prune -f --volumes || true'
}



pipeline {
    agent any

    parameters {
        string(name: 'TEST_NODE_LABEL', defaultValue: 'MI300X_BANFF', description: 'Node or Label to launch Jenkins Job')
        string(name: 'GPU_ARCH', defaultValue: 'gfx942', description: 'GPU Architecture')
    }

    environment {
        // Define repository name and tag variables
        REPO_NAME = 'rocm/megatron-lm'
        DOCKER_TAG = ''
        CONTAINER_NAME = "megatron-lm-container"
        DOCKER_BUILD_ARGS = ""
        DOCKER_RUN_ARGS = "-v \$(pwd):/workspace/Megatron-LM/output --workdir /workspace/Megatron-LM \
        --entrypoint /workspace/Megatron-LM/run_unit_tests.sh"
        DOCKER_RUN_CMD= "docker run --rm -t --network host -u root --group-add video --cap-add=SYS_PTRACE \
        --cap-add SYS_ADMIN --device /dev/fuse --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
        --ipc=host --device=/dev/kfd --device=/dev/dri"
    }
    stages {
        stage('Build Docker Image') {
            agent { label 'build-only' }
            steps {
                show_node_info()
                script {
                    env.DOCKER_BUILD_ARGS = "--build-arg PYTORCH_ROCM_ARCH_OVERRIDE=${params.GPU_ARCH}"
                    sh "docker build  -f Dockerfile_rocm.ci -t ${REPO_NAME}  ${DOCKER_BUILD_ARGS} ."
                    }
                }
            }

        stage('Tag Docker Image') {
            steps {
                script {
                    // Get the image SHA
                    def imageSha = sh(script: "docker images --format '{{.ID}}' ${DOCKER_IMAGE}:latest", returnStdout: true).trim()

                    // Get the short SHA (first 12 characters)
                    env.DOCKER_TAG = imageSha.take(12)

                    // Tag the image with the short SHA
                    sh "docker tag ${IMAGE_NAME}:latest ${REPO_NAME}:${DOCKER_TAG}"
                }
            }
        }
        stage('Fail subsequent stages') {
            steps {
                script {
                   sh 'exit 1'
                }
            }
        }
        stage('Push Docker Image') {
            steps {
                script {
                    // Log in to Docker Hub (or another registry) if needed
                    // withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                    //     sh 'echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin'
                    // }
                    // Push the tagged Docker image to the repository
                    sh "docker push ${REPO_NAME}:${DOCKER_TAG}"
                }
            }
        }

        stage('Run Unit Tests') {
            agent {
                node {
                    label "${params.TEST_NODE_LABEL}"
                    }
                }
            steps {
                script {
                     // Pull the Docker image from the repository on the test node
                    sh "docker pull ${REPO_NAME}:${DOCKER_TAG}"

                    wrap([$class: 'AnsiColorBuildWrapper', 'colorMapName': 'xterm']) {
                        sh "${DOCKER_RUN_CMD} ${DOCKER_RUN_ARGS} --name ${CONTAINER_NAME} ${REPO_NAME}:${DOCKER_TAG}"
                    }
                }
            }
        }
    }

    post {
        always {
            //Cleanup
            archiveArtifacts artifacts: 'test_report.csv'
            script {
                sh "docker rmi ${IMAGE_NAME} || true"
                sh "docker rmi ${REPO_NAME}:${DOCKER_TAG} || true"
            }
        }
    }
}
