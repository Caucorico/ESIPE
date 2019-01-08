INSERT INTO carte_fidelite ( idmag, numcli) VALUES (
	(SELECT idmag FROM magasin WHERE nom='La cabale des cables' AND ville='marseille'),
	(SELECT numcli FROM client WHERE nom='Gallois' AND prenom='Noémie' )
);

UPDATE carte_fidelite
SET point_fid=5
WHERE numcli=(
	SELECT numcli FROM client WHERE nom='Gallois' AND prenom='Noémie' 
)
AND idmag=(
	SELECT idmag FROM magasin WHERE nom='La cabale des cables' AND ville='marseille'
);

INSERT INTO carte_fidelite ( date_creation, point_fid, idmag, numcli ) VALUES (
	'2017-01-01',
	75,
	9,
	37
);

INSERT INTO carte_fidelite ( date_creation, point_fid, idmag, numcli ) VALUES (
	'2017-01-01',
	40,
	9,
	38
);

UPDATE carte_fidelite
SET point_fid=(point_fid+(point_fid/10))
WHERE date_creation < '2018-12-25';

DELETE FROM carte_fidelite
WHERE numcli=(
	SELECT numcli FROM client WHERE nom='Gallois' AND prenom='Noémie'
);
