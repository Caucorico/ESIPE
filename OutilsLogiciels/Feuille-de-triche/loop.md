# FEUILLE DE TRICHE - Les boucles

## Les boucles while
En python, il est possible de créer des boucles __while__.
Elle prennent en argument une condition et s'arrête lorsque cette condition devient fausse.

Exemple :
```python
i = 0
# Afficher les nombre de 1 à 9 :
while i < 10:
  print(i)
  i += 1
```

## Les boucles for

En python, il n'existe que des boucles __foreach__. C'est à dire qui itère sur une liste d'éléments.

Exemple :
```python
liste = [0, 1, 3, 8]

# Afficher les éléments de la liste :
for element in liste:
  print(element)
```

## La fonction range

Il est également possible de faire des boucles for `normale`, pour se faire, il faut utiliser la fonction __range()__.

- Il est possible de créer une liste simple partant de __0__ à __n__ :
```python
# Afficher les chiffres de 0 à 3 :
for i in range(4):
   print(i)
```
> Attention, la borne supérieure est exclue !

- Il est également possible de changer la borne inférieure de la liste :
```python
# Afficher les chiffres de 4 à 8 :
for i in range(4, 9):
   print(i)
```

- On peut aussi changer le `pas` (`step`) en anglais, c'est à dire l'écart entre chaque élément de la liste :
```python
# Afficher les 10 premiers nombres pair :
for i in range(0, 10, 2):
   print(i)
```

## Itérer avec les index

Il est intéressant dans certains cas de connaître l'index des élément que l'on parcours dans une liste. Python permet de faire ça en utilisant la fonction __enumerate__.

Exemple :
```python
liste = ["alice", "bob"]

# Afficher tous les couples (clef, valeur) de la liste :
for people in enumerate(liste):
  print("( " + people[0] + ", " + people[1] + " )")

```
