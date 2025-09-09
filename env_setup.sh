#!/bin/bash

# in order to be able to run the script first run "chmod +x make_env.sh"

echo "Downloading pyenv's dependencies ..."

sudo apt update -y
sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y

echo "Downloading pyenv ..."

curl https://pyenv.run | bash

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

echo "Downloading python 3.12 ..."

pyenv install 3.12

pyenv local 3.12

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv sync

echo "Install pre-commit ..."
pre-commit s

echo "Configure Git User email ..."
git config --global user.email "alexandrosliapates@gmail.com"
