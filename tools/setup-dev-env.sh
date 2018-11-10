#!/bin/bash

set -ex

apt-get update

# installing git lfs
CONF_URL="https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh"
if ! hash git-lfs 2>/dev/null; then
    curl -s $CONF_URL > get-gitlfs.sh
    bash get-gitlfs.sh
    apt-get install git-lfs
    git lfs install
    rm get-gitlfs.sh
fi

apt-get install -y python-dev python-virtualenv python-setuptools python-wheel python-pip
apt-get install -y python-m2crypto
apt-get install -y zlib1g-dev
apt-get install -y libboost1.54-all-dev
apt-get install -y libtool m4 automake
