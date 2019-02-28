#!/bin/bash
uri=$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1]))" $1)
echo $uri
wget -O/dev/null http://localhost:1234/synthesize?text=$uri
echo wget finished
##pactl load-module module-native-protocol-tcp auth-ip-acl=127.0.0.1
##ssh -R 9999:127.0.0.1:4713 you@remotehost
#paplay -s 127.0.0.1:9999 /home/znzhang1/speaker/Tacotron-2/temp.wav
