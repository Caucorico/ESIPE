#!/bin/bash
users_count=`who -m | wc -l`
process_count=`ps -A | wc -l`
echo "Nb users connectés: $users_count"
echo "Nb de processus: $process_count"

