# FEUILLE DE TRICHE - Les conditions

___

En python, il est bien entendu possible de faire des conditions.

# Les blocs if/then/else :
> Il n'est pas nécessaire d'ajouter les parenthèses autour d'une condition en python.

### Conditions basiques :
```python
if x == 2:
  print(x)
  # some code
else:
  print(x)
  # Other code
```
### Le else if :
```python
if x == 2:
  print(x)
elif x == 3:
  print(x)
else:
  print(x)
```

### Les opérateurs de comparaison :

| Opérateur | Définition                   |
|-----------|------------------------------|
| ==        | égalité                      |
| !=        | différence                   |
| <         | inférieur à                  |
| >         | supérieur à                  |
| <=        | inférieur ou égal            |
| >=        | supérieur ou égal            |

### Les opérateurs logiques :

| Opérateur | Définition                                                           |
|-----------|----------------------------------------------------------------------|
| and       | et logique |
| or        | Renvoie `True` si une des comparaisons vaut `True`                   |
| not       | Renvoie `True` si la comparaison vaut `False` (et inversement)       |
