import org.apache.commons.io.FilenameUtils
import groovy.json.JsonOutput


def clean_up_docker_images() {
    // Check if the images exist before attempting to remove them
    def imageExists = sh(script: "docker images -q ${env.DOCKER_IMAGE}", returnStdout: true).trim()
    if (imageExists) {
        sh "docker rmi ${env.DOCKER_IMAGE}"
    }
}

def clean_workspace() {
    if (env.WORKSPACE) {
        sh "sudo find ${WORKSPACE} -mindepth 1 -maxdepth 1 -exec rm -rf '{}' \\;"
    }
}

pipeline {
    agent {
        label 'build-only'
    }

    options {
        disableConcurrentBuilds(abortPrevious: true)
    }

    parameters {
        string(name: 'TEST_NODE_LABEL', defaultValue: 'linux-mi325-8', description: 'Node or Label to launch Jenkins Job')
        string(name: 'GPU_ARCH', defaultValue: 'gfx942', description: 'GPU Architecture')
    }

    environment {
        REPO_NAME = 'rocm/megatron-lm-private'
        CONTAINER_NAME = "megatron-lm-container"
        DOCKER_RUN_ARGS = "-v \$(pwd):/workspace/Megatron-LM/output --workdir /workspace/Megatron-LM \
        --entrypoint /workspace/Megatron-LM/run_unit_tests.sh"
        DOCKER_RUN_CMD = "docker run --rm --network=host -u root --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size 128G"
    }

    stages {
        stage('Set GitHub Status: Pending') {
            steps {
                script {
                    // Overall pipeline status set to pending as soon as we start
                    githubNotify context: 'jenkins/pr',
                                 credentialsId: 'rocm-repo-management-api',
                                 status: 'PENDING',
                                 description: "Jenkins build #${env.BUILD_NUMBER} is running on ${params.TEST_NODE_LABEL} (GPU: ${params.GPU_ARCH})",
                                 targetUrl: env.BUILD_URL
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    // Mark the build stage as pending
                    githubNotify context: 'jenkins/pr:build',
                                 credentialsId: 'rocm-repo-management-api',
                                 status: 'PENDING',
                                 description: "Building Docker image for GPU_ARCH=${params.GPU_ARCH}",
                                 targetUrl: env.BUILD_URL

                    // Generate a unique UUID for the Docker image name
                    def uuid = sh(script: 'uuidgen', returnStdout: true).trim()
                    env.DOCKER_IMAGE = "${REPO_NAME}:${uuid}"

                    // Build Docker image
                    sh "docker build --no-cache -f Dockerfile_rocm.ci --build-arg PYTORCH_ROCM_ARCH_OVERRIDE=${params.GPU_ARCH} -t ${env.DOCKER_IMAGE} ."

                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        sh "docker login -u ${DOCKER_USERNAME} -p ${DOCKER_PASSWORD}"
                        sh "docker push ${env.DOCKER_IMAGE}"  
                    }
                }
            }
            post {
                success {
                    script {
                        githubNotify context: 'jenkins/pr:build',
                                     credentialsId: 'rocm-repo-management-api',
                                     status: 'SUCCESS',
                                     description: "Docker image build succeeded for GPU_ARCH=${params.GPU_ARCH}",
                                     targetUrl: env.BUILD_URL
                    }
                }
                failure {
                    script {
                        githubNotify context: 'jenkins/pr:build',
                                     credentialsId: 'rocm-repo-management-api',
                                     status: 'FAILURE',
                                     description: "Docker image build failed for GPU_ARCH=${params.GPU_ARCH}",
                                     targetUrl: env.BUILD_URL
                    }
                }
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
                    // Mark the test stage as pending
                    githubNotify context: 'jenkins/pr:tests',
                                 credentialsId: 'rocm-repo-management-api',
                                 status: 'PENDING',
                                 description: "Running unit tests on ${params.TEST_NODE_LABEL} (GPU: ${params.GPU_ARCH})",
                                 targetUrl: env.BUILD_URL

                    // Pull the Docker image from the repository on the test node
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        sh "docker login -u ${DOCKER_USERNAME} -p ${DOCKER_PASSWORD}"
                        sh "docker pull ${env.DOCKER_IMAGE}"
                    }

                    wrap([$class: 'AnsiColorBuildWrapper', 'colorMapName': 'xterm']) {
                        sh "${DOCKER_RUN_CMD} ${DOCKER_RUN_ARGS} --name ${env.CONTAINER_NAME} ${env.DOCKER_IMAGE}"
                    }
                }
            }
            post {
                success {
                    script {
                        githubNotify context: 'jenkins/pr:tests',
                                     credentialsId: 'rocm-repo-management-api',
                                     status: 'SUCCESS',
                                     description: "Unit tests passed on ${params.TEST_NODE_LABEL} (GPU: ${params.GPU_ARCH})",
                                     targetUrl: env.BUILD_URL
                    }
                }
                failure {
                    script {
                        githubNotify context: 'jenkins/pr:tests',
                                     credentialsId: 'rocm-repo-management-api',
                                     status: 'FAILURE',
                                     description: "Unit tests failed on ${params.TEST_NODE_LABEL} (GPU: ${params.GPU_ARCH})",
                                     targetUrl: env.BUILD_URL
                    }
                }
                always {
                    // Publish JUnit results for Jenkins Test Result Analyzer and test trends
                    script {
                        junit testResults: 'junit_report_*.xml', allowEmptyResults: true

                        archiveArtifacts artifacts: 'unified_test_report.csv', allowEmptyArchive: true
                        clean_up_docker_images()
                        clean_workspace()
                    }
                }
            }
        }
    }

    post {
        success {
            script {
                githubNotify context: 'jenkins/pr',
                             credentialsId: 'rocm-repo-management-api',
                             status: 'SUCCESS',
                             description: "Jenkins build #${env.BUILD_NUMBER} passed on ${params.TEST_NODE_LABEL} (GPU: ${params.GPU_ARCH})",
                             targetUrl: env.BUILD_URL
            }
        }
        failure {
            script {
                githubNotify context: 'jenkins/pr',
                             credentialsId: 'rocm-repo-management-api',
                             status: 'FAILURE',
                             description: "Jenkins build #${env.BUILD_NUMBER} failed on ${params.TEST_NODE_LABEL} (GPU: ${params.GPU_ARCH})",
                             targetUrl: env.BUILD_URL
            }
        }
        aborted {
            script {
                githubNotify context: 'jenkins/pr',
                             credentialsId: 'rocm-repo-management-api',
                             status: 'ERROR',
                             description: "Jenkins build #${env.BUILD_NUMBER} was aborted on ${params.TEST_NODE_LABEL} (GPU: ${params.GPU_ARCH})",
                             targetUrl: env.BUILD_URL
            }
        }
    }
}
