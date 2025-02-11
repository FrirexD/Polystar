# PolyStar : A Quelle Star Ressembles-tu ?

## Description
PolyStar est une application web interactive permettant aux utilisateurs de se prendre en photo ou de charger une image, afin de découvrir la célébrité à laquelle ils ressemblent le plus. En utilisant des modèles de machine learning avancés, l’application analyse les caractéristiques faciales pour trouver la meilleure correspondance. De plus, un système de morphing génère un visage combinant celui de l’utilisateur et celui de la célébrité correspondante.

---

## Fonctionnalités
- Chargement ou capture de photos utilisateur.
- Correspondance avec la célébrité la plus similaire à l'aide d'un modèle CNN.
- Fusion des traits de l’utilisateur et de la célébrité grâce à un modèle de diffusion pour la génération d'images.
- Interface utilisateur intuitive pour visualiser les résultats.

---

## Technologies Utilisées
### Backend
- **Python** avec **Flask** pour la gestion des API et du backend.
- [**InsightFace**](https://insightface.ai) une librairie de reconnaissance faciale open-source

### Frontend
- **React.js** pour l’interface utilisateur dynamique.
- **CSS** pour le design et la mise en page.

### Conteneurisation
- **Docker** pour isoler l'environnement de développement et de production.

---

## Prérequis
- Python 3.9+
- Node.js 14+
- Docker 

---

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votre-repo/polystar.git
   cd polystar
   ```

Lancer l'applcation :
```bash
docker-compose -f docker-compose-app up --build
```

2. Accédez à l’application sur : [http://localhost:3030](http://localhost:3030)

---

## Utilisation
- Chargez une photo ou prenez une photo via la caméra.
- Cliquez sur le bouton pour analyser l'image.
- L'application affiche la célébrité qui vous ressemble le plus avec son score à côté de vous

