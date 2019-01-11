-- Le nombre total de factures stockées dans la base de données.
SELECT COUNT(idfac) FROM facture;


--La liste des noms de magasins, avec pour chacun le nombre de villes où ils sont implantés. Le nombre maximum à trouver est 4.
SELECT nom, COUNT(DISTINCT ville)
FROM magasin
GROUP BY nom;

--La liste des numéros et noms de clients avec pour chacun le nombre de factures qui le concernent. Attention à ne pas perdre les clients qui n’ont jamais rien acheté.
SELECT client.numcli, client.nom, client.prenom, COUNT(facture.idfac) AS nbrFactures
FROM client LEFT JOIN facture
ON client.numcli = facture.numcli
GROUP BY client.numcli, client.nom, client.prenom
ORDER BY nbrFactures;

-- Le prix moyen, minimum et maximum d’un bureau à Paris.
SELECT AVG(stocke.prixunit) AS prix_moyen, MIN(stocke.prixunit) AS prix_min, MAX(stocke.prixunit)
FROM produit
INNER JOIN stocke
ON produit.idpro = stocke.idpro
INNER JOIN magasin
ON stocke.idmag = magasin.idmag
WHERE produit.libelle = 'bureau' AND magasin.ville = 'paris';

-- La liste des meilleurs prix pour chaque libellé de produit.
SELECT produit.libelle, MIN(stocke.prixunit)
FROM produit
LEFT JOIN stocke
ON produit.idpro = stocke.idpro
GROUP BY produit.libelle;

-- a liste de toutes les factures, avec pour chacune le nom complet du client qui l’a contractée et son montant total, triées par montant décroissant.
-- La facture la plus chère coûte 1712.45 euros.
SELECT facture.idfac, client.nom, client.prenom, SUM(contient.prixunit*contient.quantite) AS montant
FROM facture
INNER JOIN contient
ON facture.idfac = contient.idfac
INNER JOIN client
ON facture.numcli = client.numcli
GROUP BY facture.idfac, client.numcli
ORDER BY montant DESC;

-- La liste des magasins qui vendent au moins 20 produits de libellés différents.
SELECT magasin.idmag, magasin.nom
FROM magasin
INNER JOIN stocke
USING(idmag)
INNER JOIN produit
USING(idpro)
GROUP BY magasin.idmag
HAVING COUNT(DISTINCT produit.libelle) >= 20;