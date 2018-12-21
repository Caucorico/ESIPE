#Requete 1 :

SELECT nom, ville, tel
FROM magasin;

#Requete 2 : 

SELECT nom, prenom
FROM client;

#Requete 3 :

SELECT prenom || ' ' || nom AS nom_complet
FROM client;

# Requete 4 :

SELECT DISTINCT CONCAT(prenom, ' ', nom) AS nom_complet
FROM client;

#Requete 5 : La liste des villes où il y a un magasin.

SELECT DISTINCT ville
from magasin;

#Requete 6 : La liste de toutes les informations sur les produits qui sont des souris.

SELECT *
FROM produit
WHERE libelle='souris';

#Requete 7 : La liste des identifiants et libellés de produits dont la couleur n’est pas renseignée.

SELECT idpro
FROM produit
WHERE couleur IS NULL;

# Requete 8 : La liste des libellés de produits qui sont des cables (quel que soit le type de cable)

SELECT libelle
FROM produit
WHERE libelle LIKE '%cable%';

# Requete 9 : La liste des numéros de clients qui ont acheté des produits dans le magasin 17

SELECT numcli
FROM client
NATURAL JOIN facture
WHERE idmag = 17;

# Requete 10 : La liste des noms et prénoms des clients qui ont acheté des produits dans le magasin 17

SELECT client.nom, client.prenom
FROM client
NATURAL JOIN facture
WHERE idmag = 17;

# Requete 11 : La liste des magasins (idmag, nom, ville) qui ont des souris en stock

SELECT DISTINCT magasin.idmag, magasin.nom, magasin.ville
FROM magasin
NATURAL JOIN stocke
NATURAL JOIN produit
WHERE produit.libelle = 'souris';

# Requete 12 : Le nom et la ville du magasin le moins cher pour acheter une souris verte, avec le prix du produit.

SELECT magasin.nom, magasin.ville, stocke.prixunit
FROM magasin
NATURAL JOIN stocke
NATURAL JOIN produit
WHERE produit.libelle = 'souris'
	AND produit.couleur = 'vert'
ORDER BY stocke.prixunit ASC
LIMIT 1;

# Requete 13 : La liste des identifiants et noms de produits qui ont été vendus à plus de 200 euros avec le prix de vente et le nom de l’acheteur

SELECT produit.idpro, produit.libelle, contient.prixunit, client.nom AS acheteur
FROM facture
INNER JOIN contient ON contient.idfac = facture.idfac
INNER JOIN produit ON contient.idpro = produit.idpro
INNER JOIN client ON client.numcli = facture.numcli
WHERE contient.prixunit > 200;

# Requete 14 : La liste des identifiants, libellés et prix de produits que l’on peut trouver à moins de 5 euros en magasin, triés par prix croissants

SELECT DISTINCT produit.idpro, produit.libelle, stocke.prixunit
FROM produit
INNER JOIN stocke ON produit.idpro = stocke.idpro
INNER JOIN magasin ON magasin.idmag = stocke.idmag
WHERE stocke.prixunit < 5
ORDER BY stocke.prixunit ASC;

# Requete 15 : La liste des libellés de produits qui existent à la fois en bleu et en jaune

SELECT produit.libelle
FROM produit
WHERE couleur = 'bleu'
INTERSECT
SELECT produit.libelle
FROM produit
WHERE couleur = 'jaune';

# Requete 16 : La liste des numéros, noms et prénoms des clients qui ont acheté un bureau

SELECT DISTINCT client.numcli, client.nom, client.prenom
FROM client
INNER JOIN facture ON facture.numcli = client.numcli
INNER JOIN contient ON facture.idfac = contient.idfac
INNER JOIN produit ON contient.idpro = produit.idpro
WHERE produit.libelle = 'bureau';

# Requete 17 : La liste des numéros, noms et prénoms des clients qui n’ont jamais acheté de bureau

SELECT client.numcli, client.nom, client.prenom
FROM client
EXCEPT
SELECT client.numcli, client.nom, client.prenom
FROM client
INNER JOIN facture ON facture.numcli = client.numcli
INNER JOIN contient ON facture.idfac = contient.idfac
INNER JOIN produit ON contient.idpro = produit.idpro
WHERE produit.libelle = 'bureau';