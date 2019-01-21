CREATE TABLE abonne (
	id SERIAL PRIMARY KEY,
	nom VARCHAR(20) NOT NULL,
	prenom VARCHAR(30) NOT NULL,
	num_tel VARCHAR(20) NOT NULL,
	mail VARCHAR(50) NOT NULL,
	mdp VARCHAR(255) NOT NULL,
	id_carte INT DEFAULT NULL
);

ALTER TABLE abonne
ADD CONSTRAINT UC_ID_CARTE_ABONEE UNIQUE (id_carte);

CREATE TABLE ville (
	id SERIAL PRIMARY KEY,
	nom VARCHAR(50) NOT NULL
);

CREATE TABLE station (
	code_station SERIAL PRIMARY KEY,
	adresse VARCHAR(50) NOT NULL,
	ville INT NOT NULL,
	FOREIGN KEY (ville) REFERENCES ville(id)
);

CREATE TABLE vehicule (
	id SERIAL PRIMARY KEY,
	statut VARCHAR(30) NOT NULL DEFAULT 'fonctionnel'
);

CREATE TABLE velo (
	id SERIAL PRIMARY KEY,
	duree_max_emprunt INTERVAL NOT NULL DEFAULT '1',
	id_vehicule INT NOT NULL,
	FOREIGN KEY (id_vehicule) REFERENCES vehicule(id)
);

CREATE TABLE voiture (
	immatriculation VARCHAR(20) NOT NULL,
	duree_max_emprunt INTERVAL NOT NULL DEFAULT '3',
	id_vehicule INT NOT NULL,
	FOREIGN KEY (id_vehicule) REFERENCES vehicule(id)
);

CREATE TABLE emplacement (
	id SERIAL PRIMARY KEY,
	type SMALLINT NOT NULL,
	disponible SMALLINT NOT NULL,
	station INT NOT NULL,
	vehicule INT DEFAULT NULL,
	FOREIGN KEY (station) REFERENCES station(code_station),
	FOREIGN KEY (vehicule) REFERENCES vehicule(id)
);

CREATE TABLE reservation_emplacement (
	id SERIAL PRIMARY KEY,
	date_reservation TIMESTAMP NOT NULL DEFAULT NOW(),
	duree_reservation INTERVAL NOT NULL,
	fin_reservation TIMESTAMP DEFAULT NULL,
	reserveur INT NOT NULL,
	fini SMALLINT NOT NULL DEFAULT 0,
	emplacement INT NOT NULL,
	FOREIGN KEY (reserveur) REFERENCES abonne(id),
	FOREIGN KEY (emplacement) REFERENCES emplacement(id)
);

CREATE TABLE reservation_vehicule (
	id SERIAL PRIMARY KEY,
	date_reservation TIMESTAMP NOT NULL DEFAULT NOW(),
	duree_reservation INTERVAL NOT NULL,
	fin_reservation TIMESTAMP DEFAULT NULL,
	reserveur INT NOT NULL,
	fini SMALLINT NOT NULL DEFAULT 0,
	vehicule INT NOT NULL,
	FOREIGN KEY (reserveur) REFERENCES abonne(id),
	FOREIGN KEY (vehicule) REFERENCES vehicule(id)
);

CREATE TABLE emprunt (
	id SERIAL PRIMARY KEY,
	debut_emprunt TIMESTAMP NOT NULL DEFAULT NOW(),
	duree_emprunt INTERVAL NOT NULL,
	fin_emprunt TIMESTAMP NOT NULL,
	emprunteur INT NOT NULL,
	fini SMALLINT NOT NULL DEFAULT 0,
	vehicule INT NOT NULL,
	FOREIGN KEY (emprunteur) REFERENCES abonne(id),
	FOREIGN KEY (vehicule) REFERENCES vehicule(id)
);