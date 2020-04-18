# FEUILLE DE TRICHE - Les dictionnaires

Il est possible de créer des dictionnaires en python.

## Créer un dictionnaire

Pour créer un dictionnaire, vous pouvez utiliser les méthodes suivantes :
```python
dict = {} # Empty dictionnary

# Feeded doctionnary.
dict = {"key1": "value1", "key2" => "value2"}
```
> Attention, pour les clefs, seuls les objets hashable peuvent servir !

## Accéder au valeur d'un dictionnaires

Pour accéder au valeur d'un dictionnaire, il suffit de spécifier la clef et le nom du dictionnaire :
```python
dict = {"key1": "value1", "key2" => "value2"}

# Access key1
x = dict["key1"]
print(x) # => value1
```

Cependant, si la clef n'existe pas, il vaut mieux utiliser la méthode __get()__. La méthode __get()__ renvoie __None__ si la clef n'existe pas.

```python
dict = {"key1": "value1"}

# Access key1
x = dict.get("key1")
print(x) # => value1

# Access an invalid key
x = dict.get("invalid_key")
print(x) # => None
```

Il est possible d'ajouter une valeur par défaut dans le __get()__. Cela permet d'éviter d'avoir None dans certains cas :
```python
dict = {"key1": "value1"}

# Access an invalid key
x = dict.get("invalid_key", "key2")
print(x) # => key2
```

## Vérifier si une clef existe dans le dictionnaire

Poir vérifier si une clef existe dans le dictionnaire, on peut utiliser le mot clef __in__.

Exemple :
```python
dict = {"key1": "value1"}

x = "key1" in dict
print(x) # => True

x = "invalid_key" in dict
print(x) # => False
```

## Ajouter des entrées au dictionnaire

Il existe plusieurs méthodes pour ajouter des éléments dans un dictionnaire :
 - Utiliser la méthode par défaut :
 ```python
 dict = {}
 dict["key1"] = "value1"

 x = dict["key1"]
 print(x) # => value1
 ```

 - Ajouter l'élément seulement s'il n'existe pas en utilisant __setdefault()__ :
 ```python
 dict = { "key1": "value1" }

 dict.setdefault("key2", "value2")
 x = dict["key2"]
 print(x) # => "value2"

 dict.setdefault("key1", "value666")
 x = dict["key1"]
 print(x) # => value1
 ```

 - La dernière méthode est en utilisant la méthode __update__. On passe en argument un autre dictionnaire qui sera fusionné :
 ```python
 dict = {}

 dict.update({"key1": "value1"})
 x = dict["key1"]
 print(x) # => value1
 ```


## Supprimer un élément du dictionnaire :

Pour supprimer un élément du dictionnaire, on peut utiliser le mit clef __del__ :
```python
dict = {"key1": "value1"}

del dict["key1"]
x = dict.get("key1")
print(x) # => None
```
