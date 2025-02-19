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
                    bat '''
                        if not exist venv python -m venv venv
                    '''
                }
            }
        }

        stage('Install Dependencies') {
            steps {
                script {
                    echo "Installing dependencies..."
                    bat '''
                        call venv\\Scripts\\activate
                        pip install --upgrade pip
                        pip install -r requirements.txt
                    '''
                }
            }
        }

        stage('Code Quality Checks') {
            steps {
                script {
                    echo 'Running code quality checks...'
                    bat '''
                        call venv\\Scripts\\activate
                        python -m flake8 . --exclude=venv --max-line-length=100
                    '''
                }
            }
        }

        stage('Prepare Data') {
            steps {
                script {
                    echo "Preparing dataset..."
                    bat '''
                        call venv\\Scripts\\activate
                        make prepare
                    '''
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    echo "Training model..."
                    bat '''
                        call venv\\Scripts\\activate
                        make train
                    '''
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    echo "Running tests..."
                    bat '''
                        call venv\\Scripts\\activate
                        make test
                    '''
                }
            }
        }

        stage('Clean Up') {
            steps {
                script {
                    echo "Deactivating virtual environment..."
                    bat '''
                        call venv\\Scripts\\deactivate.bat
                    '''
                }
            }
        }
    }
}
