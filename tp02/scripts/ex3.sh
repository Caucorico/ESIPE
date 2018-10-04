#!/bin/bash

filename="/tmp/bilan"
touch $filename
date > $filename
echo "" >> $filename
uname -a >> $filename
echo "" >> $filename
who -u >> $filename
echo "" >> $filename
ps -A >> $filename
