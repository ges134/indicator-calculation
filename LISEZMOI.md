# Programme de calcul d'une nouvelle méthode de pondération intégrée objective subjective qui utilise l'analyse par composantes principales pour corriger les biais de corrélation entre les critères dans la prise de décision en développement durable

Ce programme de calcul interroge la base de données Eurostat pour des jeux de données d'indicateurs et fusionne ceux-ci en un jeu de donnée. Puis, il calcule certains éléments de l'approche intégrée objective-subjective détaillée dans l'article. Les éléments suivants sont calculés:

- Analyse par composante principale (ACP) avec les degrés d'indépendances.
- Intervalles de confiance de l'ACP
- Poids subjectifs

Ce logiciel et son code source sont donnés comme matériel supplémentaire de l'article _A novel integrated objective-subjective weighting method using principal component analysis to correct correlation bias among criteria in sustainability decision making_ avec la référence complète suivante:

{{ RÉFÉRENCE À AJOUTER }}

Les données utilisées dans l'article, qui sont calculées à l'aide de ce programme, sont disponibles comme jeu de données dans la référence suivante:

Anglehart-Nunes, Jessy; Glaus, Mathias, 2025, "Dataset of A novel integrated objective-subjective weighting method using principal component analysis to correct correlation bias among criteria in sustainability decision making", https://doi.org/10.5683/SP3/NYKQXH, Borealis, VERSION PROVISOIRE, UNF:6:IWrDTTuFjH5uetnHcQTp/Q== [fileUNF]

Ce LISEZMOI donne de l'information sur l'utilisation du programme. Elle présente un guide de prise en main et d'utilisation. Elle présente également l'exécution du programme et les sous-programmes créés.

## Prise en main

Le projet a été développé avec [Python 3.12](https://www.python.org/) et [Pipenv](https://pypi.org/project/pipenv/). Ces dépendances devraient être installées avant l’utilisation et le développement du programme.

Ce programme nécessite un fichier `config.json` dans le dossier `data/`. Un exemple de fichier `config.template.json` est donné comme point de départ pour être utilisé dans ce programme. Les fichiers `ProgramConfiguration` sont aussi donnés pour des fins de réplication.

Le fichier de configuration est une liste d'objets. Chaque objet a la structure suivante:

```json
{
  "id": "internal id",
  "code": "eurostat code",
  "social": 0,
  "environmental": 0,
  "economic": 0
  // Liste de clés-valeurs pour les dimensions.
}
```

Où:

- `id` (**obligatoire**): Un identifiant pour cet indicateur. L’identifiant est le code interne dans la grille d’analyse de l’indicateur.
- `code` (**obligatoire**): Le code Eurostat pour récupérer les données des indicateurs.
- `social` (**obligatoire**): Le score d’association, entre 0 et 3, pour le pilier social de cet indicateur.
- `environmental` (**obligatoire**): Le score d’association, entre 0 et 3, pour le piller environnemental de cet indicateur.
- `economic` (**obligatoire**): Le score d’association, entre 0 et 3, pour le pilier économique de cet indicateur.
- La liste de clés valeurs est la liste des dimensions nécessaires pour filtrer les données et extraire les données superflues. Cette liste doit inclure une entrée pour chaque dimension multiple dans le jeu de données d’Eurostat. Par exemple, si un jeu de données Eurostat rapporte un indicateur sur plusieurs unités de mesure, une clé-valeur doit être précisée pour indiquer quelle unité doit être sélectionnée par le programme. Le bouton _customize your dataset_ donne la liste des dimensions pour l’indicateur. Les dimensions _Time_, _Time Frequency_ et _Geopolitical entity (reporting)_ devraient être omises. Le programme s’occupe de traiter ces derniers et s’ils sont absents, le programme ne fusionnera pas le jeu de données. Finalement, si la dimension rapporte une seule valeur dans Eurostat, elle peut être omise de la configuration.

Les scores sont sur une échelle de 0 à 3 avec une signification particulière. Les scores peuvent être interprétés de la façon suivante:

- 0: Cet indicateur ne mentionne pas le pilier de développement durable.
- 1: Cet indicateur mentionne le pilier de développement durable.
- 2: Cet indicateur est l'un des accents principaux de cet indicateur.
- 3: Cet indicateur est le seul accent principal de cet indicateur.

Pour rouler ce projet, les dépendances doivent être installées avec la commande suivante:

```sh
pipenv install
```

## Utilisation

Le programme peut être invoqué d'une ligne de commandes et doit être invoqué à partir du code source. La commande est la suivante:

```sh
pipenv run py main.py
```

## Fonctionnement du programme

Le programme traite chaque indicateur dans le fichier `config.json`. Chaque indicateur reçoit le traitement suivant:

1. Le jeu de données est chargé de la base de données Eurostat et converti en un `Dataframe`.
1. Le jeu de données est testé pour les conditions de fusion. Un jeu de données peut être fusionné s'il y a un horodatage annuel, une entité géopolitique et un temps.
1. Le jeu de données est filtré pour avoir une seule dimension selon le fichier de configuration. Par exemple, s'il y a plusieurs unités de mesure, le programme utilisera celui spécifié dans le fichier de configuration.
1. Avec le format approprié, le programme lit chaque rangée et met à jour le jeu de données fusionnées.

Le jeu de données ainsi créé est converti en un jeu de données prêt pour l'ACP. Le programme fait ensuite le traitement suivant:

1. Calculer l’ACP sur les indicateurs.
1. Calculer les degrés d’indépendances sur les indicateurs.

Un degré d'indépendance est une valeur située entre 0 et 1 qui montre comment une paire d'indicateurs est dépendante l'une de l'autre. La valeur 0 indique une dépendance totale alors que 1 indique une indépendance totale.

Ensuite, le programme calcule les intervalles de confiance basée sur la méthode de Bootstrap. Cela implique les étapes suivantes:

1. Normaliser les données des indicateurs qui ne suivent pas la loi normale.
1. Créer les échantillons de Bootstrap avec leur ACP.
1. Créer les échantillons de Jacknife avec leur ACP.
1. Calculer les intervalles de confiances avec le niveau de confiance 0,01 et 0,05.

Le programme réutilise ensuite le fichier de configuration pour simuler une procédure hiérarchique d'analyse (AHP) et ensuite calculer les poids subjectifs. Le programme effectue les traitements suivants:

1. Convertir les scores en échelle de Likert
1. Calculer les matrices de comparaison selon l’échelle de Likert
1. Appliquer l’AHP pour chaque pilier et pour les piller avec l’analyse de la consistance.

L'échelle de Likert peut être interprétée avec les règles suivantes:

- 1: L'indicateur a un score de 0 ou de 1 pour le pilier.
- 3: L'indicateur a un score de 2 pour le pilier.
- 5: L'indicateur a un score de 3 pour le pilier. Une différence de 2 points est observée pour les autres piliers.
- 7: L'indicateur a un score de 3 pour le pilier. Une différence de 1 point est observée pour les autres piliers.
- 9: L'indicateur est un match parfait avec le pilier.

## Données sauvegardées

Le programme sauvegarde plusieurs fichiers de résultats. Ils sont détaillés ci-bas:

- `angles.csv`: L’angle entre chaque indicateur.
- `bootstraped-dataset.csv` : Les observations de Bootstraped tirées par le programme.
- `confidence-intervals-01.csv` : La valeur des bornes des intervalles de confiance pour le niveau de confiance de 0,01.
- `confidence-intervals-05.csv` : La valeur des bornes des intervalles de confiance pour le niveau de confiance de 0,05.
- `consistency.csv`: L’analyse de la consistance par piller (piliers inclus dans l’analyse).
- `economic-comparison-matrix`: La matrice de comparaison pour le pilier économique.
- `eigen-values.csv`: Les valeurs propres de chaque composant principal.
- `eigen-vectors.csv`: Les vecteurs propres de chaque composant principal.
- `empiric-eigen-vectors.csv`: Les vecteurs propres de chaque composant principal avec la normalisation des données pour Bootstrap.
- `environmental-comparison-matrix`: La matrice de comparaison pour le pilier environnemental.
- `explained-variance.csv`: La variance expliquée de chaque composant principal.
- `independance_degree.csv`: Le degré d’indépendance de chaque indicateur.
- `jacknifed-dataset.csv` : Les observations de Jacknife construites par le programme.
- `merged.csv`: Les données fusionnées des indicateurs.
- `scores.csv` : Les scores de chaque indicateur dans l’échelle de Likert.
- `social-comparison-matrix.csv` : La matrice de comparaison pour le pilier social.
- `weights.csv` : Les poids de chaque indicateur pour chaque pilier et avec les poids finaux.

Les angles et les degrés d'indépendances sont sous un format diagonal. Toute valeur en dessous de la diagonale n'est pas utilisée. De plus, bien que ce programme sauvegarde des diagrammes de contributions, ceux présentés dans l'article sont disponibles comme `notebooks` dans le répertoire `notebooks/`.

## Sensibilité pour l'année d'observation

Un sous-programme applique l'ACP et calcul les degrés d'indépendances pour trois années observées avec des données complètes: la première, celle du milieu et la dernière. Pour utiliser ce sous-programme, le fichier de configuration doit être fourni. La commande suivante peut rouler le programme.

```sh
pipenv run py years.py
```

Le programme fera le traitement suivant:

1. Charger la configuration
1. Obtenir les données pour les indicateurs
1. Retirer les années avec des observations incomplètes
1. Calculer l’ACP et les degrés d’indépendances
1. Sauvegarder les résultats dans des fichiers.

Les fichiers sauvegardés détiennent le format ANNEE-eigen-vectors.csv et ANNEE-independance-degree.csv, qui ont le même format que le programme principal, mais les résultats ont été calculés avec les observations d’une ANNEE.

## Veille des changements des données

Un sous-programme vérifie pour des changements dans les données des indicateurs dans la base de données d'Eurostat. Pour surveiller de tels changements, le fichier de configuration et un fichier de données fusionnées de référence doivent être fournis. Le fichier de données fusionnées de référence est le fichier `merged.csv` associé avec la configuration du programme. Le fichier doit être nommé `reference.csv`. La commande suivante peut rouler le sous-programme:

```sh
pipenv run py monitor.py
```

Le programme accomplit les étapes suivantes:

1. Charger les données de référence et exécuter la fusion des données.
2. Créer un `DataFrame` de comparaison
3. Sauvegarder le `DataFrame`.

Le fichier sauvegardé se nomme `monitored.csv` et suit le [format de Pandas de la fonction `compare`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.compare.html). En somme, si le fichier est vide, aucun changement n’a été reporté. Le `DataFrame` rapporte seulement les changements et est donc un sous-ensemble du fichier de fusion. Conséquemment, seulement les colonnes avec des indicateurs modifiés apparaitront dans le jeu de données de veille. Similairement, seulement des rangées avec des valeurs modifiées apparaitront dans le jeu de données de veille. Les colonnes représentent les indicateurs avec `reference` et `new`. Le premier fait référence au fichier de référence fourni pour l’exécution du programme tandis que la seconde fait référence aux données trouvées par l’exécution du programme. Le numéro de rangée apparait dans la première colonne pour faciliter le suivi des changements. `NaN` peut apparaitre dans le fichier pour indiquer qu’aucun changement n’a été observé pour cette rangée et cet indicateur.
