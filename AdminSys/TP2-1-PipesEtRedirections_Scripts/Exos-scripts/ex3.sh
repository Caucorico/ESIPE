#!/bin/bash

echo "Date : " > /tmp/bilan
date >> /tmp/bilan
echo "nom de la machine : " >> /tmp/bilan
uname -a >> /tmp/bilan
echo "liste des utilisateurs connectés : " >> /tmp/bilan
# Attention, ici, si le script est executer depuis un terminal dans interface graphique,
# l'utilisateur courant apparaitra 2 fois
who >> /tmp/bilan
echo "Liste des processus : " >> /tmp/bilan
ps -e >> /tmp/bilan