-- La liste des lieux, avec pour chacun le prix moyen d’entrée des soirées qui y sont organisées.

SELECT lieu, AVG(entree)
FROM soiree
GROUP BY lieu;

-- La liste des prénoms des personnes déguisées en Père Noël dans une soirée à Paris.

SELECT DISTINCT prenom
FROM participe
NATURAL JOIN personne
NATURAL JOIN soiree
WHERE avatar='Père Noël' AND lieu='paris';

-- Les soirées organisées dans une ville dont le nom commence par ‘T’.

SELECT ids
FROM soiree
WHERE lieu LIKE 't%';

-- Les villes où sont organisées des soirées pouvant accueillir au moins 145 personnes.

SELECT DISTINCT lieu
FROM soiree
WHERE nbmax >= 145;

-- Les surnoms et prénoms des personnes de 18 ans qui sont venues sans déguisement à une soirée à Lyon.

SELECT surnom, prenom
FROM personne
NATURAL JOIN participe
NATURAL JOIN soiree
WHERE age=18 AND avatar IS NULL AND lieu='lyon';

-- Les villes dans lesquelles au moins 1000 euros ont été dépensés en entrées à des soirées.

SELECT lieu
FROM soiree
NATURAL JOIN participe
GROUP BY lieu, entree
HAVING 1000 <= entree*COUNT(surnom);

-- La liste des personnes qui ont participé à au moins trois soirées à Marseille, triée par nombre de soirées décroissant.

SELECT surnom, prenom, COUNT(ids) AS nbrsoiree
FROM personne
NATURAL JOIN participe
NATURAL JOIN soiree
WHERE lieu='marseille'
GROUP BY surnom
HAVING COUNT(ids) >= 3
ORDER BY nbrsoiree DESC;

-- L’âge moyen des organisateurs de soirées.

SELECT AVG(age)
FROM personne
WHERE surnom IN (
	SELECT DISTINCT surnom
	FROM personne
	INNER JOIN soiree
	ON soiree.organisateur = personne.surnom
);

-- Les soirées où tous les participants ont le même avatar.

SELECT DISTINCT ids , lieu, date
FROM soiree
NATURAL JOIN participe AS p1
WHERE avatar = ALL(
	SELECT avatar
	FROM participe AS p2
	WHERE p1.ids =  p2.ids
);

-- Les soirées où tous les participants ont un avatar différent

SELECT DISTINCT ids
FROM soiree
NATURAL JOIN participe AS p1
WHERE (
	SELECT COUNT(*)
	FROM participe AS p2
	WHERE p1.ids = p2.ids	
) = (
	SELECT COUNT(DISTINCT avatar)
	FROM participe AS p3
	WHERE p1.ids = p3.ids
);