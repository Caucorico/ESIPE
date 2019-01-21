--
--	Utilisez le script magasin_exam.sql afin de créer la base de données qui sera utilisée pour les questions suivantes.
--	Prenez le temps de vous familiariser avec les différentes tables et leur contenu avant de commencer à répondre aux questions.
--	Pour chaque question, donnez la requête SQL permettant d'obtenir le résultat demandé, et ajoutez en commentaire le nombre de lignes renvoyé par cette requête.
--  Soignez la présentation (indentation, nom des variables, etc.) de vos requêtes. La lisibilité de votre travail sera prise en compte.
--	A la fin du TP, déposez votre fichier sur elearning dans l'espace dédié à votre groupe.
--  Pensez à enregistrer régulièrement votre travail afin de ne pas le perdre en cas de panne.
--


--1. La liste de tous les produits avec tous leurs attributs.

SELECT *
FROM produit;
-- Cette requ^ete renvoie 100 lignes.

--2. La liste des villes où il y a un magasin, sans doublons.

SELECT DISTINCT ville
FROM magasin;
-- Cette requ^ete renvoie 20 lignes.

--3. La liste des identifiants des produits qui sont des souris blanches.

SELECT idpro
FROM produit
WHERE libelle='souris'
	AND couleur='blanc';
-- Cette requ^ete renvoie 1 ligne.

--4. La liste des identifiants de produits dont un magasin a plus de 1000 exemplaires en stock.

SELECT DISTINCT idpro
FROM produit
NATURAL JOIN stocke
WHERE quantite > 1000;
-- Cette requ^ete renvoie 35 lignes.

--5. La liste des cartes de fidélité (numcarte) qui ont été créées en 2017.

--SELECT numcarte
--FROM fidelite
--WHERE datecreation > '2016';


--6. La liste des magasins (idmag, nom) qui ont au moins un bureau en stock.

SELECT DISTINCT idmag, nom
FROM magasin
NATURAL JOIN stocke
NATURAL JOIN produit
WHERE libelle='bureau';
-- Cette requ^ete renvoie 46 lignes.

--7. Les numéros et noms des clients qui ont fait un achat dans un magasin qui n'est pas dans la ville où ils habitent.

SELECT numcli, nom
FROM client
NATURAL JOIN facture
NATURAL JOIN magasin
WHERE client.ville != magasin.ville;
-- Cette requ^ete renvoie 0 ligne.


--8. Les identifiants des téléphones dont on connait la couleur.

SELECT idpro
FROM produit
WHERE libelle='téléphone'
	AND couleur IS NOT NULL;
-- Cette requ^ete renvoie 1 ligne.

--9. Les noms de famille que l'on trouve à la fois à Marseille et à Reims.

-- todo : remplacer par une req^ete avec INTERSECT
SELECT DISTINCT nom
FROM client
WHERE ville='reims'
INTERSECT
SELECT DISTINCT nom
FROM client
WHERE ville='marseille';
-- Cette requ^ete renvoie 1 ligne.

--10. Les villes dans lesquelles il y a des magasins mais pas de clients.

SELECT DISTINCT ville
FROM magasin
EXCEPT
SELECT DISTINCT ville
FROM client;
-- Cette requ^ete renvoie une ligne.

--11. Le prix moyen et le prix le plus bas d'un bureau.

SELECT AVG(prixunit) AS prix_moyen, MIN(prixunit) AS prix_min
FROM produit
NATURAL JOIN stocke
WHERE libelle='bureau'
GROUP BY libelle;
-- Cette requ^ete renvoie une ligne

--12. La liste des villes où il y a au moins un client, avec le nombre de clients qui s'y trouvent.

SELECT ville, COUNT(numcli) AS nombre_client
FROM client
GROUP BY ville;
-- Cette requ^ete renvoie 23 lignes.

--13. La liste des clients (numcli, prenom, nom) qui ont passé au moins 10 commandes, triée par nombre de commandes décroissant.

SELECT numcli, prenom, nom
FROM client
NATURAL JOIN facture
GROUP BY numcli, prenom, nom
HAVING COUNT(idfac) >= 10
ORDER BY COUNT(idfac) DESC;
-- Cette requ^ete renvoie 6 lignes.

--14. La liste des magasins (idmag, nom) avec pour chacun l'argent total qu'il a encaissé.

CREATE OR REPLACE VIEW valeur_facture AS
SELECT SUM(prixunit*quantite) AS prix_fac, idfac, idmag, numcli
FROM facture
NATURAL JOIN contient
GROUP BY idfac;

SELECT idmag, nom, SUM(prix_fac) AS total_encaisse
FROM magasin
NATURAL JOIN valeur_facture
GROUP BY idmag;
-- Cette requ^ete renvoie 50 ligne.


--15. La liste des paires de clients différents (numcli1, prenom1, numcli2, prenom2) qui ont fait un achat le même jour dans le même magasin.

SELECT client1.numcli AS numcli1, client1.prenom AS prenom1, client2.numcli AS numcli2, client2.prenom AS numcli2
FROM client AS client1, client AS client2
NATURAL JOIN facture
WHERE client1.numcli != client2.numcli
	AND 

--16. Les clients (numcli, prenom, nom) qui ont assez d'argent sur une de leurs cartes de fidélité pour acheter le produit le plus cher du magasin d'où provient la carte.

CREATE OR REPLACE VIEW total_fidelite AS
SELECT numcli, SUM(points) AS total_points
FROM client
NATURAL JOIN fidelite
GROUP BY numcli;

SELECT numcli, prenom, nom
FROM client
NATURAL JOIN total_fidelite AS tf
WHERE total_points >= (
	SELECT MAX(prixunit)
	FROM stocke
	NATURAL JOIN magasin AS m1
	WHERE tf.idmag = m1.idmag
);

--17. La liste des clients (numcli) avec pour chacun le montant de la facture la moins chère qu'il a payée.

SELECT numcli, MIN(prix_fac) AS moins_chere_facture
FROM client
NATURAL JOIN valeur_facture
GROUP BY numcli;
-- Cette requete renvoie 200 lignes.


--18. Le client (numcli, prenom, nom) qui a le plus de points de fidelité, toutes cartes confondues.

SELECT client.numcli, prenom, nom
FROM client
NATURAL JOIN total_fidelite
WHERE total_points = (
	SELECT MAX(total_points)
	FROM client
	NATURAL JOIN total_fidelite
);
-- Cette fonction renvoie 1 ligne.


--19. Les clients (numcli, prenom, nom) dont toutes les factures s'élèvent à plus de 750 euros.

SELECT numcli, prenom, nom
FROM client
NATURAL JOIN valeur_facture
GROUP BY numcli
HAVING MIN(prix_fac) > 750;
-- Cette requ^ete renvoie 4 lignes.

--20. Les magasins qui ont au moins un produit de chaque libellé en stock.



