# 🔬 Expérimentation – Impact des Paramètres sur la Précision (Accuracy)

Ce document décrit une série d’expériences visant à analyser l’impact de diverses **modifications** (prétraitement, choix de paramètres, etc.) sur la **précision du modèle de régression logistique** utilisé pour la classification d’images thoraciques.

## ⚙️ Réglages de Base (Baseline)

| Paramètre        | Valeur                                |
| ----------------- | ------------------------------------- |
| Modèle           | `LogisticRegression(max_iter=1000)` |
| Taille des images | `(400, 400)`                        |
| Mode              | `Grayscale`, aplatie                |
| Normalisation     | Pixels entre 0 et 1                   |
| Jeu de test       | 20% des données, stratifié          |

**Accuracy de base :** 86%

---

## 🧪 Batterie de Tests

### 🔁 1. Variation du prétraitement

| ID | Modification              | Description                                   | Résultat (Accuracy) |
| -- | ------------------------- | --------------------------------------------- | -------------------- |
| V1 | `image_size=(200, 200)` | Réduction de la taille pour accélérer      | 85%                  |
| V2 | `image_size=(100, 100)` | Compression agressive, possible perte d'infos | 87%                  |

---

### ⚙️ 2. Modifications du modèle

Parametre de base: `image_size=(100, 100)`

| ID | Modification      | Description                                            | Résultat (Accuracy) |
| -- | ----------------- | ------------------------------------------------------ | -------------------- |
| M1 | `max_iter=2000` | Plus d’itérations pour convergence                   | 87%                  |
| M2 | `solver='saga'` | Meilleur pour grands datasets                          | 88%                  |
| M3 | `penalty='l1'`  | Régularisation L1 (sparse solutions)                  | 83%                  |
| M4 | `C=0.1`         | Plus forte régularisation (moins de surapprentissage) | 86%                  |
| M5 | `C=10.0`        | Moins de régularisation (plus de flexibilité)        | 87%                  |

---

## 📊 Résumé à compléter après chaque test

| Test ID | Accuracy | Observations                                               |
| ------- | -------- | ---------------------------------------------------------- |
| V1      | 85%      | Réduction légère du temps d'entraînement, perte minime |
| V2      | 87%      | Accuracy dégradée, trop peu d'info dans les images       |
| M4      | 86%      | Régularisation trop forte → sous-apprentissage           |

---

## 📌 Recommandations

- Conserver les images en `400x400` ou réduire à `200x200` max.
- Tester `C` entre `0.1` et `10.0` pour trouver un bon compromis.

---

🛠️ Code type pour un test

```python
model = LogisticRegression(
    solver='saga',
    penalty='l1',
    C=0.5,
    max_iter=2000,
)
```
