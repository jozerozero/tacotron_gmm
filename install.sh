#!/bin/bash
apt-get update

apt-get install python3-dev
apt-get install python-pyaudio
apt-get install python3-pyaudio
apt-get install portaudio19-dev
apt-get install locales
locale-gen en_US.UTF-8
update-locale

echo -e 'export PYTHONIOENCODING=utf-8\nexport LANG="en_US.utf8"\nexport LC_ALL="en_US.utf8"\nexport LC_CTYPE="en_US.utf8"' >> ~/.bashrc

#use python3
pip install -r requirements.txt
