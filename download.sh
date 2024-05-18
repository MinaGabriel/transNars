#!/bin/sh

BASEDIR=`pwd`

if [ ! -d "$BASEDIR" ]; then
    mkdir "$BASEDIR"
fi

# toy (testing dataset)
if [ ! -d "$BASEDIR/toy" ]; then
    echo Downloading toy
    cd $BASEDIR
    curl -O https://web.informatik.uni-mannheim.de/pi1/kge-datasets/toy.tar.gz
    tar xvf toy.tar.gz
else
    echo toy already present
fi