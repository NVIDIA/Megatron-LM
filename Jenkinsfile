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

DOCKER_IMAGE = "megatron-lm"
CONTAINER_NAME = "megatron-lm-container"
DOCKER_BUILD_ARGS = "--build-arg PYTORCH_ROCM_ARCH_OVERRIDE=gfx90a"
DOCKER_RUN_ARGS = "-v \$(pwd):/workspace/Megatron-LM/output --workdir /workspace/Megatron-LM --entrypoint /workspace/Megatron-LM/run_unit_tests.sh"

DOCKER_RUN_CMD= "docker run --rm -t --network host -u root --group-add video --cap-add=SYS_PTRACE --cap-add SYS_ADMIN --device /dev/fuse --security-opt seccomp=unconfined --security-opt apparmor=unconfined --ipc=host --device=/dev/kfd --device=/dev/dri"
pipeline {
    parameters {
        string(name: 'TEST_NODE_LABEL', defaultValue: 'MI250', description: 'Node or Label to launch Jenkins Job')
    }

    agent {node {label "${params.TEST_NODE_LABEL}"}}

    stages {
        stage('Build Docker Image') {
            steps {
                show_node_info()
                script {
                    sh "docker build  -f Dockerfile_rocm_ci -t ${DOCKER_IMAGE}  ${DOCKER_BUILD_ARGS} ."
                    }
                }
            }

        stage('Run Docker Container') {
            steps {
                script {
                    wrap([$class: 'AnsiColorBuildWrapper', 'colorMapName': 'xterm']) {
                        sh "${DOCKER_RUN_CMD} ${DOCKER_RUN_ARGS} --name ${CONTAINER_NAME} ${DOCKER_IMAGE} "
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
                sh "docker rmi ${DOCKER_IMAGE}"
            }
        }
    }
}
