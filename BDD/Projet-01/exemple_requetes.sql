-- Requete effectuer lors de la connexion :

SELECT *
FROM abonne
WHERE mail=$mail AND mdp=$mdp;

-- Requete a effectuer lors de l'inscription :

INSERT INTO abonne(nom, prenom, num_tel, mail, mdp, id_carte)
VALUES ( $nom, $prenom, $num_tel, $mail, $mdp, $id_carte);

-- Trouver les velo disponible sur la station 1

SELECT DISTINCT id_vehicule
FROM velo
INNER JOIN vehicule
ON velo.id_vehicule = vehicule.id
INNER JOIN emplacement
ON vehicule.id = emplacement.vehicule
INNER JOIN station
ON station.id = emplacement.station
WHERE station.id = 1;

-- Le nombre de velos disponible sur la station 1

SELECT COUNT(*)
FROM velo
INNER JOIN vehicule
ON velo.id_vehicule = vehicule.id
INNER JOIN emplacement
ON vehicule.id = emplacement.vehicule
INNER JOIN station
ON station.code_station = emplacement.station
WHERE station.code_station = 1
GROUP BY velo.id;
