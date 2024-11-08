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

def clean_up_docker_container() {
    sh 'docker ps -a || true' // "|| true" suppresses errors
    sh 'docker kill $(docker ps -q) || true'
}

//makes sure multiple builds are not triggered for branch indexing
def resetbuild() {
    if(currentBuild.getBuildCauses().toString().contains('BranchIndexingCause')) {
        def milestonesList = []
        def build = currentBuild

        while(build != null) {
            if(build.getBuildCauses().toString().contains('BranchIndexingCause')) {
                milestonesList.add(0, build.number)
            }
            build = build.previousBuildInProgress
        }

        for (buildNum in milestonesList) {
            milestone(buildNum)
        }
    }
}

pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                echo 'Building..'
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
            }
        }
        stage('Deploy') {
            steps {
                show_node_info()
            }
        }
    }
}
