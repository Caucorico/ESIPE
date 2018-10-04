#!/bin/bash
if [ $# -lt 1 ]
then
   echo "Usage: $0 <filename>"
   exit 1
fi

filename=$1

file_type=`file $filename --mime-type`
if [[ $filename = *".sh" ]]
then
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
else
   echo "C'est pas un script bash"
fi
