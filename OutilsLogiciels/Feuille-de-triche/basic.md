# FEUILLE DE TRICHE - Générale
___
# L'indentation :

En python, les "blocs de données", (la partie entre accolades dans la plupart des autres langages), est reconnue grâce à l'indentation des lignes dans le fichier.

Exemple :

```python
# fonction
def fonction():
  # début contenu fonction (décalé d'une tabulation)
  x = 2
  print(x)
  if x == 2:
    # début du contenu du if (décalé de deux tabulation car déjà dans une fonction)
    print(x)
    # fin if
  #fin fonction

# autre code...
```

> Attention, toute erreur dans le nombre de tabulation avant une ligne peut la faire changer de bloc !
___
# Début d'un bloc :

En règle générale, il ne suffit pas d'ajouter la tabulation pour indiquer le début d'un nouveau bloc. Il faut utiliser le symbole "***:*** "

### Exemples :

Déclarer une fonction et son contenu :
```python
def function():
  # Contenu de la fonction
```

Déclarer une condition :
```python
if x > 3:
  # Contenu du if
else:
  # Contenu du else
```

___

# Indiquer à un système Linux que le fichier est du python :
Par défaut, sur le système Linux, il est nécessaire d'indiquer avec quel langages/outils exécuter notre script.
```Bash
python3.7 mon_script.py
```

Or, dans de nombreux cas, nous ne souhaitons pas avoir à faire le choix de l'outil à utiliser.
> De plus, si nous récupérons un script sur le web, nous ne savons pas forcément quel python utiliser.

Pour ce faire, il est possible d'indiquer au début du script quel outil utiliser :
```python
#!/bin/python3

def fonction():
  print("In fonction()")
  # etc...
```
