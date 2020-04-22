#!/bin/bash
Date=$(date +%Y%m%d%H%M)
if [ ! -d "logs" ];then
mkdir logs
fi

nohup python -m app.preprocess \
>> ./logs/console_$Date.log 2>&1 &