#!/bin/bash

echo "Liste des partitions : " > BILAN
echo "" >> BILAN
df -a >> BILAN
echo "" >> BILAN
echo "--------------------------------------------------" >> BILAN
echo "" >> BILAN
echo "Liste des point de montages : " >> BILAN
echo "" >> BILAN
mount >> BILAN
echo "" >> BILAN
echo "--------------------------------------------------" >> BILAN
echo "" >> BILAN
echo "Le SE fonctionne depuis : " >> BILAN
uptime -p >> BILAN