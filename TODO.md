# TODO

## Finaliser l'implémentation de l'apprentissage

- [x] faire une sélection de features sur les patchs pour descendre à 4 features
  - [x] :exclamation: explorer les possibles algorithmes de sélection de features
  - [x] :exclamation: tester la sélection de features et voir si elle améliore le(s) modèle(s)
- [ ] améliorer les classifiers (random forest et SVM)
  - [x] :bangbang: rendre robuste la recherche d'hyperparamètres
  - [ ] :grey_exclamation: essayer d'optimiser le RF
- [ ] :interrobang: implémenter une approche par Bootstrap pour la validation
- [ ] implémenter les différents modèles suivants

Lettres | Images Complète | 4 patches | 16 patches avec sélection pour passer à 4 |
---|---|---|---|
Avec les petites lettres|:exclamation: :x:|:bangbang: :x:|:bangbang: :x:|
Sans les petites lettres|:exclamation: :x:|:bangbang: :x:|:bangbang: :x:|

## Tester le(s) modèles final(aux) sur un exemple de texte coptique

- [ ] obtenir les éléments afin de tester le texte
  - [x] :bangbang: récupérer les images d'origines
  - [ ] :bangbang: récupérer les algorithmes de décomposition d'image faits l'année dernière si disponible
  - [ ] :interrobang: si non disponible écrire un algorithme pour décomposer l'image copte en liste d'image de lettres
- [ ] tester le(s) modèle(s) sur le texte copte
  - [ ] :exclamation: écrire un script pour charger un modèle et un ensemble de lettres qui ressort une liste de charactère copte
  - [ ] :exclamation: faire un rapport résumé sur le code et les performances

## Vérifier la cohérence du projet

- [x] vérification des données du problème
  - [x] :bangbang: vérifier la taille des images d'entrées
  - [x] :bangbang: vérifier s'il existe d'autres hypothèses non-vérifiées

## Lexique

- :bangbang: priorité haute
- :exclamation: priorité normale
- :grey_exclamation: priorité basse
- :interrobang: inconnue, à définir
