#!/bin/bash
echo "Quel est votre prénom ?"
read firstname
echo "Quel est votre nom ?"
read lastname
echo "Quel est votre mois de naissance ?"
read birthday_month
echo "Quel est votre année de naissance ?"
read birthday_year
echo "Nom : $firstname $lastname Né en $birthday_month $birthday_year"
cal $birthday_month $birthday_year