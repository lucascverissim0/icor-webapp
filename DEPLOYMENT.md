# Déploiement — ICOR webapp (Streamlit Community Cloud)

Le client n'a **rien à installer** : il ouvre une URL et se connecte. Toute la
config tourne côté serveur. Aucune dépendance à la version de Python locale.

## 1. Déployer l'app

1. Aller sur **https://share.streamlit.io** et se connecter avec le compte GitHub
   propriétaire du repo (`lucascverissim0`).
2. **Create app → Deploy a public app from GitHub** :
   - Repository : `lucascverissim0/icor-webapp`
   - Branch : `main`
   - Main file path : `ui/app.py`
   - **Advanced settings → Python version : 3.12** (correspond à l'env testé).
3. Coller les **secrets** (section 2 ci-dessous) dans *Advanced settings → Secrets*.
4. **Deploy**. Au bout de 1-2 min, une URL stable `https://<nom>.streamlit.app`
   est générée → c'est le lien à envoyer au client.

À chaque `git push` sur `main`, Streamlit Cloud redéploie automatiquement.

## 2. Secrets (à coller dans le dashboard, JAMAIS dans le repo)

Format TOML, mappé sur `st.secrets` :

```toml
[users]
# 1 ligne par utilisateur. Mot de passe haché bcrypt (voir §3).
client1 = { name = "Nom du client", password = "$2b$12$....hash...." }
admin   = { name = "Admin",         password = "$2b$12$....hash...." }

[openai]
api_key = "sk-..."        # requis pour Script 1 (BEV) et Model Researcher

[serpapi]
api_key = "..."           # optionnel : améliore le seeding web du Model Researcher

# Optionnel — analytics
[posthog]
api_key = "phc_..."
host = "https://app.posthog.com"
```

## 3. Générer un mot de passe haché (bcrypt)

```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'LE_MOT_DE_PASSE', bcrypt.gensalt()).decode())"
```

Copier le `$2b$...` obtenu dans le champ `password`. Le code accepte aussi un mot
de passe en clair, mais le hash bcrypt est recommandé en production.

## 4. Régénérer les données avec une vraie clé OpenAI

Le `data/passenger_car_data.xlsx` committé est une **baseline** générée sans clé
OpenAI (scores BEV approximatifs). Pour des scores BEV exacts : une fois l'app en
ligne avec la clé OpenAI dans les secrets, cliquer **« Run backend (Script 1) »**
dans la barre latérale. ⚠️ Le système de fichiers de Streamlit Cloud est éphémère :
les données régénérées y vivent jusqu'au prochain redéploiement/veille. Pour figer
une version, télécharger le classeur (bouton « Download workbook ») et le re-committer.

## Limites du free tier

- L'app se met en veille après inactivité (réveil ~30 s à la 1re visite).
- ~1 Go de RAM ; le Model Researcher (2-10 min/modèle) peut approcher le timeout
  de 420 s sur les modèles lourds.
- URL publique mais protégée par le login.
- Pour un usage commercial privé/persistant sans veille → migrer vers un conteneur
  Docker sur Render/Azure (évolution future).

## Temporary Codespaces preview (development branch only)

The generation-aware React/FastAPI preview is separate from the legacy Streamlit
deployment above. It is not production hosting and must never be created from or
merged into `main`. Use only `development/windshield-demand-platform`, keep port 8000
private through bootstrap and owner smoke testing, and follow
`docs/DEVELOPMENT.md#github-codespaces-preview`. The application authentication gate
must be verified before temporarily selecting **Port visibility -> Public**, and the
port must be returned to **Private** when review ends or any check fails.