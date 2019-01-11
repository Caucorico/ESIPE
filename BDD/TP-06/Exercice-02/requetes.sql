-- La liste des identifiants et noms de magasins qui ne vendent pas de bureaux.

SELECT *
FROM magasin
WHERE idmag NOT IN (
	SELECT idmag
	FROM magasin
	INNER JOIN stocke
	USING(idmag)
	INNER JOIN produit
	USING(idpro)
	WHERE produit.libelle = 'bureau'
);

-- La liste des magasins dont tous les produits sont à moins de 100 euros.

SELECT *
FROM magasin
WHERE 100 > (
	SELECT MAX(stocke.prixunit)
	FROM stocke
	WHERE stocke.idmag = magasin.idmag
	GROUP BY idmag
);

-- La liste des produits qu’aucun client n’a acheté.

SELECT *
FROM produit
WHERE idpro NOT IN (
	SELECT DISTINCT idpro
	FROM produit
	INNER JOIN contient
	USING(idpro)
);

-- La liste des identifiants et libellés de produits qui ont été vendus au moins 40% plus cher que leur prix moyen sur le marché.

SELECT produit.idpro, produit.libelle
FROM produit
INNER JOIN contient
USING(idpro)
WHERE contient.prixunit > 1.4*(
	SELECT AVG(stocke.prixunit)
	FROM stocke
	WHERE stocke.idpro = produit.idpro
);

-- La liste des noms et prénoms de clients qui ont acheté un produit au moins 20 euros plus cher que son prix moyen, avec les libellés des produits en question.

SELECT client.nom, client.prenom
FROM client
WHERE client.numcli IN (
	SELECT client.numcli
	FROM client
	INNER JOIN facture
	USING(numcli)
	INNER JOIN contient
	USING(numfac)
)