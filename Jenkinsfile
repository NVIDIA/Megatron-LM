import org.apache.commons.io.FilenameUtils
import groovy.json.JsonOutput


def clean_up_docker_images() {
    // Check if the images exist before attempting to remove them
    def imageExists = sh(script: "docker images -q ${env.DOCKER_IMAGE}", returnStdout: true).trim()
    if (imageExists) {
        sh "docker rmi ${env.DOCKER_IMAGE}"
    }
}

def clean_docker_build_cache() {
    sh 'docker system prune -f --volumes || true'
}

pipeline {
    agent {
        label 'build-only'
    }

    parameters {
        string(name: 'TEST_NODE_LABEL', defaultValue: 'MI300X_BANFF', description: 'Node or Label to launch Jenkins Job')
        string(name: 'GPU_ARCH', defaultValue: 'gfx942', description: 'GPU Architecture')
    }

    environment {
        REPO_NAME = 'rocm/megatron-lm-private'
        CONTAINER_NAME = "megatron-lm-container"
        DOCKER_RUN_ARGS = "-v \$(pwd):/workspace/Megatron-LM/output --workdir /workspace/Megatron-LM \
        --entrypoint /workspace/Megatron-LM/run_unit_tests.sh"
        DOCKER_RUN_CMD = "docker run --rm -t --network host -u root --group-add video --cap-add=SYS_PTRACE \
        --cap-add SYS_ADMIN --device /dev/fuse --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
        --ipc=host --device=/dev/kfd --device=/dev/dri"
    }

    stages {
        stage('Build Docker Image') {
            steps {
                clean_docker_build_cache()
                script {

                    // Generate a unique UUID for the Docker image name
                    def uuid = sh(script: 'uuidgen', returnStdout: true).trim()
                    env.DOCKER_IMAGE = "${REPO_NAME}:${uuid}"

                    // Build Docker image
                    sh "docker build --no-cache -f Dockerfile_rocm.ci --build-arg PYTORCH_ROCM_ARCH_OVERRIDE=${params.GPU_ARCH} -t ${env.DOCKER_IMAGE} ."

                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        sh "docker push ${env.DOCKER_IMAGE}"  
                    }
                }
            }
            post {
                always {
                    clean_up_docker_images()
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
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        sh "docker pull ${env.DOCKER_IMAGE}"
                    }

                    wrap([$class: 'AnsiColorBuildWrapper', 'colorMapName': 'xterm']) {
                        sh "${DOCKER_RUN_CMD} ${DOCKER_RUN_ARGS} --name ${env.CONTAINER_NAME} ${env.DOCKER_IMAGE}"
                    }
                }
            }
            post {
                always {
                // Archive test results
                script {
                    archiveArtifacts artifacts: 'test_report.csv', allowEmptyArchive: true
                    clean_up_docker_images()
                    }
                }
            }
        }
    }
}
