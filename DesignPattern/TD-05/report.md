#### Quel principe SOLID n'est pas implémenté par la librairie stp ?

Cette libraire ne respecte pas l'open-close principle. Il n'y a aucun moyen de dire : "Si c'est une Hello je fais ça etc..."

#### Comment pourrait-on résoudre le problème en utilisant le polymorphisme ? Est-ce que c'est envisageable dans cette situation ? 

On pourrait créer une méthode d'interface **process** qui exécute le code de la commande.
Malheureusement, cette méthode n'est pas applicable dans cette solution. En effet, la responsabilité de créer les classe Cmd
reviendrait à l'utilisateur de la librairie ! Or, justement, cette librairie est censée fournir 4 commandes par défaut.

#### Quel patron permet d'ajouter des nouvelles opérations aux classes implémentant l'interface STPCommand sans modifier ces classes ?

Pour ce faire, il faut implémenter le design pattern du visiteur.