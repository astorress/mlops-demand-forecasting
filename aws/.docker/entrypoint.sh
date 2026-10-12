#!/bin/bash
# Se ejecuta una vez desde el postCreateCommand del devcontainer.

COLOR_GREEN=$(tput setaf 2)
COLOR_BLUE=$(tput setaf 4)
COLOR_MAGENTA=$(tput setaf 5)
COLOR_DEFAULT=$(tput sgr0)

EXEC_PATH=$(dirname $(readlink -f "$0"))
HOME_PATH=$EXEC_PATH/..
STEP=1
STEP_COUNT=4

echo "${COLOR_MAGENTA}${STEP}/$STEP_COUNT:${COLOR_GREEN} Configuring Git.${COLOR_DEFAULT}"
git config --global --add safe.directory /app
let STEP++

cd $HOME_PATH
if [ ! -d "$HOME_PATH/node_modules" ]; then
    echo "${COLOR_MAGENTA}${STEP}/$STEP_COUNT:${COLOR_GREEN} Installing ${COLOR_BLUE}$HOME_PATH/package.json${COLOR_GREEN} dependencies.${COLOR_DEFAULT}"
    npm install
else
    echo "${COLOR_MAGENTA}${STEP}/$STEP_COUNT:${COLOR_GREEN} Folder ${COLOR_BLUE}$HOME_PATH/node_modules${COLOR_GREEN} found! Skipping npm installation.${COLOR_DEFAULT}"
fi
let STEP++

echo "${COLOR_MAGENTA}${STEP}/$STEP_COUNT:${COLOR_GREEN} Granting access to the host Docker daemon.${COLOR_DEFAULT}"
if [ -S /var/run/docker.sock ]; then
    sudo chmod 666 /var/run/docker.sock
    docker version --format 'INFO: Docker server {{.Server.Version}}'
else
    echo "WARNING: /var/run/docker.sock not mounted; npm run image:push won't work."
fi
let STEP++

echo "${COLOR_MAGENTA}${STEP}/$STEP_COUNT:${COLOR_GREEN} Configuring AWS credentials.${COLOR_DEFAULT}"
AWS_SCRIPT_PATH=$HOME_PATH/.docker/aws.sh
chmod +x $AWS_SCRIPT_PATH
$AWS_SCRIPT_PATH

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${COLOR_GREEN}INFO: AWS account: ${ACCOUNT_ID}${COLOR_DEFAULT}"
echo -e "${COLOR_GREEN}INFO: DEVELOPER=${DEVELOPER}${COLOR_DEFAULT}"
