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
- **TensorFlow** ou **PyTorch** pour les modèles de machine learning.
- **OpenCV** et **PIL** pour le traitement d’images.

### Frontend
- **React.js** pour l’interface utilisateur dynamique.
- **CSS** pour le design et la mise en page.

### Base de Données
- **SQLite** pour la gestion des données utilisateurs.

### Conteneurisation
- **Docker** pour isoler l'environnement de développement et de production.

### Modèles Utilisés
- **FaceNet** ou **ArcFace** pour la comparaison de visages.
- **StyleGAN** ou **modèle de diffusion** pour la génération d'images.

---

## Prérequis
- Python 3.9+
- Node.js 14+
- Docker (optionnel pour la conteneurisation)

---

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votre-repo/polystar.git
   cd polystar
   ```

2. Installez les dépendances backend :
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Installez les dépendances frontend :
   ```bash
   cd ../frontend
   npm install
   ```

4. Lancez le serveur Flask :
   ```bash
   cd ../backend
   python app.py
   ```

5. Lancez le serveur React :
   ```bash
   cd ../frontend
   npm start
   ```

6. Accédez à l’application sur : [http://localhost:3000](http://localhost:3000)

---

## Utilisation
- Chargez une photo ou prenez une photo via la caméra.
- Cliquez sur le bouton pour analyser l'image.
- L'application affiche la célébrité la plus proche et une image fusionnée.

