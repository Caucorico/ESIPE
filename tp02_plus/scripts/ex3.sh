#!/bin/bash
filename=$1
if test -f $filename
then
   if test -x $filename
   then
      echo "C'est déja un exécutable"
   else
      chmod u+x $filename
   fi
else
   echo "Le fichier n'existe pas"
fi

