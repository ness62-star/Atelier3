pipeline {
    agent any

    environment {
        VENV = "venv"
        PYTHON = "python"
    }

    stages {
        stage('Checkout Code') {
            steps {
                script {
                    echo "Cloning the latest code from GitHub..."
                    checkout scm
                }
            }
        }

        stage('Setup Virtual Environment') {
            steps {
                script {
                    echo "Creating and activating virtual environment..."
                    sh '''
                        if [ ! -d "$VENV" ]; then
                            $PYTHON -m venv $VENV
                        fi
                        source $VENV/bin/activate
                    '''
                }
            }
        }

        stage('Install Dependencies') {
            steps {
                script {
                    echo "Installing dependencies..."
                    sh 'make install'
                }
            }
        }

        stage('Code Quality Checks') {
            steps {
                script {
                    echo "Running code quality checks..."
                    sh 'make check'
                }
            }
        }

        stage('Prepare Data') {
            steps {
                script {
                    echo "Preparing dataset..."
                    sh 'make prepare'
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    echo "Training model..."
                    sh 'make train'
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    echo "Running tests..."
                    sh 'make test'
                }
            }
        }

        stage('Clean Up') {
            steps {
                script {
                    echo "Deactivating virtual environment..."
                    sh 'deactivate || true'
                }
            }
        }
    }
}
