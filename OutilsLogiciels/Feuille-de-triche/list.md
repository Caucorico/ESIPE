# FEUILLE DE TRICHE - Les listes

## Les liste en python :

En python, il n'existe pas de tableau à proprement parler, python utilise des __listes__.
Nous pouvons stocker une variable dans une liste :
```python
x = [] # Liste vide

x = [ 1, 2 ] # Liste contenant les éléments 1 et 2.
```

## Ajouter des éléments à une liste :

Il est possible d'ajouter un élément à la fin de la liste en utilisant la méthode __append()__ :
```python
x = [] # => La liste vide.

x.append(1)
print(x) # => La liste contenenant 1.

x.append(2)
print(x) # La liste contenant 1 et 2.
```
___
## Enlever des éléments d'une liste :
Il est possible d'enlever le dernier élément d'une liste :
```python

x = [ 1, 2 ]
print(x) # => La liste contenant 1 et 2

a = x.pop()
print(x) # => La liste contenant 1.
print(a) # => 2 (L'ancien dernier élément)

b = x.pop()
print(x) # => La liste vide.
print(b) # => 1
```

Il est possible de supprimer arbitrairement un élément d'une liste :

```python
x = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

del x[2]
print(a) # La liste contenant les valeurs [ 0, 1, 3, 4, 5, 6, 7, 8, 9 ]
```
___
## Accéder aux éléments d'une liste :

Il est possible d'accéder à un élément précis en connaissant sa position :
```python
x = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

# Récupérer l'élément d'indice 3 :
a = x[3]
print(a) # => 3
```

Il est possible d'accéder au dernier élément de la liste :
```python
x = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

# Récupérer le dernier élément :
a = x[-1]
print(a) # => 9
```

Il est possible de récupérer les éléments entre deux borne de la liste. La première valeur, avant le __:__ est la borne inférieure (incluse) et la deuxième valeur est la borne extérieure (excluse).

```python
x = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

# Récupérer les éléments entre l'indice 3 (inclus) et l'indice 6 (exclus) :
a = x[3:6]
print(a) # La liste contenant les valeurs 3, 4 et 5
```
Il est possible de récupérer les __n__ premier éléments de la liste :
```python
x = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

# Récupérer les 3 premier éléments :
a = x[:3]
print(a) # La liste contenant les valeurs 0, 1 et 2
```

Il est également possible d'omettre les __n__ premier éléments de la liste :
```python
x = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

# Omettre les 3 premier éléments :
a = x[3:]
print(a) # La liste contenant les valeurs [ 3, 4, 5, 6, 7, 8, 9 ].
```

Et il est également possible d'omettre les __n__ derniers éléments de la liste :
```python
x = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

# Omettre les 3 derniers éléments :
a = x[:-3]
print(a) # La liste contenant les valeurs [ 1, 2, 3, 4, 5, 6 ].
```

Il est possible de sélectionner un élément sur 2 :
```python
x = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

# Obtenir un élément sur deux :
a = x[::2]
print(a) # La liste contenant les valeurs [ 0, 2, 4, 6, 8 ].
```

Et, il est possible de récupérer la liste inversée :
```python
x = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

# Obtenir la liste inversée :
a = x[::-1]
print(a) # La liste contenant les valeurs [ 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 ].
```
___

## Concaténer deux listes :

Il est possible de concaténer deux liste en python :
- En utilisant la méthode __extend()__ :
```python
a = [ 1, 2, 3 ]
b = [ 4, 5, 6 ]
x = a.extend(b)
print(x) # => La liste contenant les valeurs [ 1, 2, 3, 4, 5, 6 ]
```
- En utilisant le symbole __+__ :
```python
a = [ 1, 2, 3 ]
b = [ 4, 5, 6 ]
x = a + b
print(x) # => La liste contenant les valeurs [ 1, 2, 3, 4, 5, 6 ]
```
___

## Copier une liste :

En python, faire simplement `liste1 = liste2` ne marchera pas pour copier une liste. Faire cette opération signifie seulement que __liste2__ pointe désormais sur __liste1__.

Pour copier une liste, nous pouvons donc :
- Utiliser l'accès `:` :
```python
liste = [ 1, 2, 3 ]
liste2 = liste[:]
print(liste2) # => [ 1, 2, 3 ]
print(liste == liste2) # => True
print(liste is liste2) # => False
```
- Utiliser les méthodes __copy__/__deepcopy__ :
```python
from copy import copy
liste = [ 1, 2, 3 ]
liste2 = copy(liste)
print(liste2) # => [ 1, 2, 3 ]
print(liste == liste2) # => True
print(liste is liste2) # => False
```
> Il est bien-entendu possible d'utiliser  __deepcopy__ pout copier les objets à l'intérieur de la liste également.
