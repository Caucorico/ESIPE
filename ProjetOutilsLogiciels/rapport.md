# Rapport projet Ariane

Vous trouverez dans le dossier courant les sources ainsi que les maps et les images nécessaires au fonctionnement de mon projet Ariane.

## Comment lancer le jeu ?

Comme demandé, un fichier ariane.py est présent et peut être lancé avec python3.
Je demande seulement un argument supplémentaire au lancement. Cet argument est le chemin vers le fichier de map que l'on souhaite utiliser.
Pour récapituler, pour lancer le programme, il faut faire :

> python3 ariane.py \<chemin_de_la_map\>

## Fonctionnalités

Dans mon projet, chaque fonctionnalité est utilisable via une touche. Vous en trouverez la liste ci-dessous :

| Fonctionalité | Touche | Description |
| :----------- | :------: | :------------ |
| Se déplacer vers le haut | Flèche du haut | Déplace Ariane sur la case supérieure si le mouvement est autorisé |
| Se déplacer vers le bas | Flèche du bas | Déplace Ariane sur la case inférieure si le mouvement est autorisé |
| Se déplacer vers la gauche | Flèche de gauche | Déplace Ariane sur la case à sa gauche si le mouvement est autorisé |
| Se déplacer vers la droite | Flèche de droite | Déplace Ariane sur la case à sa droite si le mouvement est autorisé |
| Vérifier si la victoire est possible | Touche `c` | Vérifie si la configuration actuelle est gagnante ou perdante et affiche le résultat |
| Afficher la solution dans le terminal | Touche `s` | Affiche dans le terminal une suite de coups permettant à Ariane de réussir à s'enfuir avec Thésée |
| Afficher la solution à l'écran | Touche `d` | Affiche à l'écran la suite de coup permettant à Ariane de s'enfuir avec Thésée. |
| Afficher l'algorithme de backtracking en action | Touche `v` | Affiche à l'écran les calculs de l'algorithme de backtracking. |

> Chaque action se fait à partir de l'état courant. La solution calculée l'est à partir de l'état courant.

## Remarque

Il semble que l'affichage visuel de la solution ne fonctionne pas. La map semble ne garde pas la même configuration. Je n'ai pas eu le temps de corriger le problème.
