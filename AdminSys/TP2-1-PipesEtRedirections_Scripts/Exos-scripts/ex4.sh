#!/bin/bash
nbusers=`who | wc -l`
nbprocess=`ps -e | wc -l`
echo "Nb users connectés : $nbusers"
echo "Nb process : $nbprocess"