#!/bin/bash
clear
echo "Nom prenom"
read name
echo "Mois de naissance"
read birthday_month
echo "Année de naissance"
read birthday_year
echo "Vous êtes $firstname $name, né(e) en $birthday_month $birthday_year"

cal $birthday_month $birthday_year
