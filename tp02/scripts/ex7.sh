#!/bin/bash
today_date=`date +%F` #Formatte la date en YYYY-MM-DD
filename="/tmp/SAV-${today_date}"
lastlog > $filename
