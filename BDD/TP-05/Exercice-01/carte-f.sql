CREATE TABLE carte_fidelite (
	idcf SERIAL NOT NULL, 
	date_creation DATETIME NOT NULL DEFAULT NOW(),
	point_fid INT NOT NULL DEFAULT 0,
	idmag INT NOT NULL,
	numcli INT NOT NULL,
	PRIMARY KEY (idcf),
	FOREIGN KEY (idmag) REFERENCES magasin(idmag),
	FOREIGN KEY (numcli) REFERENCES client(numcli)
);

ALTER TABLE carte_fidelite
ADD CONSTRAINT UC_carte_fidelite UNIQUE( idmag, numcli);