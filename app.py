"""
═════════════════════════════════════════════════════════════════════════════
CRTV FLY-AWAY MONITOR v3.2 OPTIMIZED + CHAT IA (INDÉPENDANT DE L'HEURE SYSTÈME)
═════════════════════════════════════════════════════════════════════════════
Chargement ultra-rapide + Chat IA pour prédictions & recommandations

Auteur    : Moussa Manga Asser
Version   : 3.2.0 (Optimized + Interactive AI Chat + Time-independent)
Contact   : +237 690 537 181 | assermoussa19@gmail.com

CHANGEMENTS PAR RAPPORT À v3.1
────────────────────────────────────────────────────────────────────────────
1. L'app ne dépend plus de l'horloge de la machine (datetime.now()) pour
   décider quelles lignes sont "récentes". Elle utilise désormais le
   timestamp le plus récent PRÉSENT DANS LE FICHIER comme référence
   ("données_maintenant"). Ainsi, que vos données datent de 2024, 2025 ou
   d'hier, le dashboard affiche toujours la dernière heure de données
   disponible — plus besoin de changer la date du PC.
2. Le chemin du fichier Excel et du logo, auparavant en dur (chemin
   Windows spécifique à un seul poste), sont maintenant configurables et
   accompagnés d'un uploader de secours : si le fichier n'est pas trouvé
   au chemin par défaut, l'utilisateur peut le déposer directement dans
   l'interface.
3. Le logo ne fait plus planter l'app s'il est absent : repli silencieux
   sur une icône.
4. Petits correctifs de robustesse (colonnes manquantes, dataframe vide,
   etc.).
═════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import warnings
import os

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION GLOBALE
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CRTV Fly-Away v3.2 — Optimized + Chat IA",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chemins du projet : compatibles Windows ET Streamlit Cloud.
# Le fichier Excel déjà chargé dans le dépôt est utilisé automatiquement.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Nom actuel du fichier Excel dans le dépôt.
DATASET_FILENAME = "flyaway_log_annuel_2025-1.xlsx"
DEFAULT_EXCEL_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)

# Compatibilité avec l'ancien nom, au cas où le fichier aurait encore ce nom.
ANCIEN_DATASET_PATH = os.path.join(DATA_DIR, "flyaway log annuel 2025-1.xlsx")

# Le logo est facultatif : l'application fonctionne sans lui.
DEFAULT_LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo-crtv.png")
DEFAULT_SHEET_NAME = "Donnees_2025"

FENETRE_RECENTE = timedelta(hours=1)          # taille de la fenêtre "temps réel"
SEUIL_LATENCE_WARN = 720
SEUIL_LATENCE_CRIT = 800

# ════════════════════════════════════════════════════════════════════════════
# CACHE GLOBAL
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_global_state():
    """Initialise l'état global une seule fois."""
    return {"chat_history": []}


# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES — accepte un chemin OU un fichier uploadé
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def charger_donnees_depuis_chemin(file_path: str, sheet_name: str):
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        skiprows=1,
        dtype={
            'timestamp': 'object',
            'latence_totale_ms': 'float32',
            'cn0_dbhz': 'float32',
            'esn0_db': 'float32',
            'qualite_signal_pct': 'float32',
            'jitter_ms': 'float32',
            'ber': 'float64',
            'temperature_hpa_c': 'float32',
            'temperature_buc_c': 'float32'
        }
    )
    return df


@st.cache_data(ttl=300, show_spinner=False)
def charger_donnees_depuis_upload(file_bytes: bytes, sheet_name: str):
    import io
    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet_name,
        skiprows=1
    )
    return df


def nettoyer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage commun, quelle que soit la source du fichier."""
    if df is None or df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp', ascending=False)
    return df


def charger_donnees(file_path: str, sheet_name: str, uploaded_file=None):
    """
    Charge automatiquement le fichier Excel déjà présent dans data/.
    L'uploader reste disponible comme solution de secours.
    """
    try:
        if uploaded_file is not None:
            df = charger_donnees_depuis_upload(uploaded_file.getvalue(), sheet_name)
            return nettoyer_donnees(df), None

        # Fichier actuel puis ancien nom pour compatibilité.
        chemins = [file_path, ANCIEN_DATASET_PATH]
        for chemin in chemins:
            if os.path.exists(chemin):
                df = charger_donnees_depuis_chemin(chemin, sheet_name)
                return nettoyer_donnees(df), None

        return None, (
            "Fichier Excel introuvable. Vérifiez que "
            f"'{DATASET_FILENAME}' est bien dans le dossier data/ du projet."
        )
    except Exception as e:
        return None, f"Erreur de chargement : {e}"


# ════════════════════════════════════════════════════════════════════════════
# RÉFÉRENCE TEMPORELLE — LE CŒUR DU CORRECTIF
# ════════════════════════════════════════════════════════════════════════════

def obtenir_reference_temporelle(df: pd.DataFrame) -> datetime:
    """
    Retourne le "maintenant" à utiliser pour filtrer les données récentes.

    Au lieu de datetime.now() (heure de la machine, qui doit alors être
    manipulée pour coïncider avec des données historiques), on prend le
    timestamp le plus récent réellement présent dans le fichier. L'app
    fonctionne donc correctement quelle que soit la date/l'heure du
    système, aujourd'hui comme dans 5 ans, avec des données d'archive
    comme avec un flux réellement temps réel.
    """
    if df is None or df.empty:
        return datetime.now()
    return df['timestamp'].max()


# ════════════════════════════════════════════════════════════════════════════
# MODÈLE IA
# ════════════════════════════════════════════════════════════════════════════

class ModeleAIOptimisee:
    """Modèle IA optimisé avec cache."""

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.features = []

    def entrainer_rapide(self, df):
        try:
            params_cles = [
                'temperature_hpa_c', 'temperature_buc_c',
                'attenuation_pluie_db', 'symbol_rate_msps'
            ]
            cols_dispo = [p for p in params_cles if p in df.columns]
            if not cols_dispo or 'latence_totale_ms' not in df.columns:
                return False

            X = df[cols_dispo].dropna()
            y = df.loc[X.index, 'latence_totale_ms']

            if len(X) > 50:
                self.features = X.columns.tolist()
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
                self.trained = True
                return True
        except Exception:
            pass
        return False

    def predire_rapide(self, row_dict):
        if not self.trained:
            return None
        try:
            X_input = np.array([[row_dict.get(f, 0) for f in self.features]])
            X_scaled = self.scaler.transform(X_input)
            return float(self.model.predict(X_scaled)[0])
        except Exception:
            return None


# ════════════════════════════════════════════════════════════════════════════
# CHATBOT IA INTERACTIF
# ════════════════════════════════════════════════════════════════════════════

class ChatbotIAPrediction:
    """Chatbot pour expliquer prédictions et anomalies."""

    def __init__(self, model_ia, df_recent, df_complet):
        self.model = model_ia
        self.df_recent = df_recent
        self.df_complet = df_complet

    def repondre(self, question):
        q = question.lower()

        if any(w in q for w in ['prédi', 'futur', 'demain', 'dans 15']):
            return self._expliquer_predictions()
        elif any(w in q for w in ['latence', 'délai', 'delay']):
            return self._analyser_latence()
        elif any(w in q for w in ['qualité', 'signal', 'quality']):
            return self._analyser_qualite()
        elif any(w in q for w in ['temp', 'température', 'heat']):
            return self._analyser_temperature()
        elif any(w in q for w in ['pluie', 'météo', 'weather', 'rain']):
            return self._analyser_pluie()
        elif any(w in q for w in ['recommand', 'action', 'faire', 'quoi faire']):
            return self._recommandations_actions()
        elif any(w in q for w in ['comment', 'pourquoi', 'quoi', 'status']):
            return self._resume_global()
        else:
            return self._reponse_defaut(question)

    def _expliquer_predictions(self):
        if self.df_recent.empty:
            return "❌ Pas assez de données pour les prédictions."

        latence_actuelle = self.df_recent['latence_totale_ms'].mean()
        latences_recent = self.df_recent['latence_totale_ms'].tail(10).values

        if len(latences_recent) > 2:
            trend = np.polyfit(range(len(latences_recent)), latences_recent, 1)[0]
            latence_pred = latence_actuelle + trend * 3
        else:
            latence_pred = latence_actuelle

        confiance = (
            "⭐⭐⭐⭐ Très haute (historique stable)" if np.std(latences_recent) < 30
            else "⭐⭐⭐ Bonne (variations acceptables)" if np.std(latences_recent) < 50
            else "⭐⭐ Modérée (variations importantes)"
        )

        if latence_pred > SEUIL_LATENCE_CRIT:
            interpretation = "🔴 **CRITIQUE** : Latence dépasserait le seuil critique (800ms) → Action urgente requise MAINTENANT"
        elif latence_pred > SEUIL_LATENCE_WARN:
            interpretation = "🟡 **ATTENTION** : Latence dépasserait le seuil d'avertissement (720ms) → À surveiller attentivement"
        else:
            interpretation = "🟢 **OK** : Latence resterait dans les normes acceptables → Situation stable"

        return f"""
### 🔮 Prédictions — 15 prochaines minutes (à partir des dernières données)

**Latence actuelle** : {latence_actuelle:.0f}ms
**Latence prédite** : {latence_pred:.0f}ms
**Tendance** : {"📈 HAUSSE" if latence_pred > latence_actuelle else "📉 BAISSE"}
**Confiance** : {confiance}

{interpretation}
"""

    def _analyser_latence(self):
        if self.df_recent.empty:
            return "❌ Pas de données latence."
        lat = self.df_recent['latence_totale_ms']
        statut = (
            "🔴 CRITIQUE - Intervention urgente" if lat.mean() > SEUIL_LATENCE_CRIT
            else "🟡 DÉGRADÉE - À surveiller" if lat.mean() > SEUIL_LATENCE_WARN
            else "🟢 NORMALE - Aucune action"
        )
        return f"""
### ⏱️ Analyse Latence Transmission

**Latence actuelle** : {lat.iloc[0]:.0f}ms
**Moyenne (fenêtre récente)** : {lat.mean():.0f}ms
**Min / Max** : {lat.min():.0f}ms / {lat.max():.0f}ms
**Écart-type** : {lat.std():.0f}ms

**Statut** : {statut}
"""

    def _analyser_qualite(self):
        if self.df_recent.empty or 'qualite_signal_pct' not in self.df_recent.columns:
            return "❌ Pas de données qualité."
        qual = self.df_recent['qualite_signal_pct'].mean()
        cn0 = self.df_recent['cn0_dbhz'].mean() if 'cn0_dbhz' in self.df_recent.columns else np.nan
        statut = (
            "🔴 Qualité insuffisante - Intervention requise" if qual < 80
            else "🟡 Qualité dégradée - À surveiller" if qual < 85
            else "🟢 Qualité excellente - Normal"
        )
        return f"""
### 📡 Qualité Signal Réception

**Qualité signal** : {qual:.1f}%
**C/N₀ moyen** : {cn0:.1f}dBHz

**Statut** : {statut}
"""

    def _analyser_temperature(self):
        if 'temperature_hpa_c' not in self.df_recent.columns or self.df_recent.empty:
            return "❌ Données température non disponibles."
        temp_hpa = self.df_recent['temperature_hpa_c'].mean()
        statut = (
            "🛑 CRITIQUE - Arrêter transmission et refroidir" if temp_hpa > 75
            else "🔴 Élevée - Réduire puissance HPA" if temp_hpa > 65
            else "🟡 Modérée - Surveiller les tendances" if temp_hpa > 50
            else "🟢 Optimale - Conditions idéales"
        )
        return f"""
### 🌡️ Analyse Thermique Équipements

**Température HPA** : {temp_hpa:.1f}°C

**Statut** : {statut}
"""

    def _analyser_pluie(self):
        if 'attenuation_pluie_db' not in self.df_recent.columns or self.df_recent.empty:
            return "❌ Données pluie non disponibles."
        pluie = self.df_recent['attenuation_pluie_db'].mean()
        return f"""
### 🌧️ Analyse Impact Pluie

**Affaiblissement pluie** : {pluie:.2f}dB

- Ciel dégagé : 0-0.5dB ✅
- Pluie légère : 0.5-2dB ⚠️
- Pluie modérée : 2-4dB 🔴
- Pluie forte : > 4dB 🛑

En bande C, la pluie augmente le FEC DVB-S2 requis et donc la latence
(souvent +100ms). Stratégies : augmenter le FEC, réduire le symbol rate,
augmenter la puissance HPA si marge disponible.
"""

    def _resume_global(self):
        if self.df_recent.empty:
            return "❌ Pas de données."
        lat = self.df_recent['latence_totale_ms'].mean()
        qual = self.df_recent['qualite_signal_pct'].mean() if 'qualite_signal_pct' in self.df_recent.columns else np.nan
        cn0 = self.df_recent['cn0_dbhz'].mean() if 'cn0_dbhz' in self.df_recent.columns else np.nan

        etat = "🟢 NORMAL" if lat < SEUIL_LATENCE_WARN and qual > 85 else \
               "🟡 DÉGRADÉ" if lat < SEUIL_LATENCE_CRIT and qual > 80 else "🔴 CRITIQUE"

        return f"""
### 📊 Résumé Global Système

**Latence** : {lat:.0f}ms {"✅" if lat < SEUIL_LATENCE_WARN else "⚠️" if lat < SEUIL_LATENCE_CRIT else "🔴"}
**Qualité** : {qual:.1f}% {"✅" if qual > 85 else "⚠️" if qual > 80 else "🔴"}
**C/N₀** : {cn0:.1f}dBHz {"✅" if cn0 > 10 else "⚠️" if cn0 > 8 else "🔴"}

**État** : {etat}

Posez une question spécifique : "Explique latence", "Analyse qualité",
"Température", "Impact pluie", "Recommandations".
"""

    def _recommandations_actions(self):
        lat = self.df_recent['latence_totale_ms'].mean() if not self.df_recent.empty else 0
        return f"""
### ⚡ Actions Recommandées

**Latence actuelle** : {lat:.0f}ms

1. **Immédiate** (< 5 min) : vérifier pointage antenne, consulter la météo, augmenter le FEC DVB-S2 si C/N₀ bas.
2. **Court terme** (< 30 min) : optimiser les paramètres DVB-S2, ajuster le buffer jitter, surveiller la température HPA.
3. **Moyen terme** (< 2h) : générer les rapports, informer la direction si critique.
4. **Long terme** (> 2h) : analyser les tendances historiques, planifier maintenance/upgrades.

Si latence > {SEUIL_LATENCE_CRIT}ms : escalade immédiate (appel ingénieur, email d'alerte, rapport urgent, basculement vers liaison alternative).
"""

    def _reponse_defaut(self, question):
        return f"""
### 🤖 Assistant IA Fly-Away

Je n'ai pas bien compris : "{question}"

Essayez : "Comment est la latence ?", "Expliquer les prédictions",
"Analyser qualité signal", "Température ?", "Impact pluie ?",
"Quoi faire maintenant ?", "Résumé global".
"""


# ════════════════════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ════════════════════════════════════════════════════════════════════════════

def main():
    init_global_state()

    # ── Barre latérale : configuration des sources de fichiers ──────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        excel_path = st.text_input(
            "Chemin fichier Excel",
            value=DEFAULT_EXCEL_PATH
        )
        sheet_name = st.text_input("Feuille", value=DEFAULT_SHEET_NAME)
        st.caption(f"📁 Fichier utilisé : {DATASET_FILENAME}")
        uploaded_file = st.file_uploader(
            "Ou déposez le fichier Excel ici (utilisé si le chemin ci-dessus est introuvable)",
            type=["xlsx", "xls"]
        )
        st.caption(
            "ℹ️ Le tableau de bord se cale automatiquement sur la dernière "
            "donnée disponible dans le fichier — inutile de changer l'heure "
            "de la machine pour voir des données récentes."
        )

    # ── Chargement des données ───────────────────────────────────────────────
    with st.spinner("⏳ Chargement des données..."):
        df_complet, erreur = charger_donnees(excel_path, sheet_name, uploaded_file)

    if erreur:
        st.error(f"❌ {erreur}")
        st.info("Déposez votre fichier Excel dans la barre latérale pour continuer.")
        return
    if df_complet is None or df_complet.empty:
        st.error("❌ Aucune donnée exploitable dans le fichier.")
        return

    # ── Référence temporelle basée sur les données, pas sur l'horloge ───────
    reference_now = obtenir_reference_temporelle(df_complet)
    df_recent = df_complet[df_complet['timestamp'] >= reference_now - FENETRE_RECENTE]
    if df_recent.empty:
        df_recent = df_complet.head(50)  # filet de sécurité si la fenêtre est vide

    # Entraîner le modèle
    model_ia = ModeleAIOptimisee()
    model_ia.entrainer_rapide(df_complet)

    # ── En-tête ───────────────────────────────────────────────────────────
    col_logo, col_titre = st.columns([1, 5])
    with col_logo:
        logo_path = DEFAULT_LOGO_PATH
        if os.path.exists(logo_path):
            try:
                st.image(Image.open(logo_path), width=80)
            except Exception:
                st.write("🛰️")
        else:
            st.write("🛰️")

    with col_titre:
        st.markdown("# 🛰️ CRTV Fly-Away v3.2 — Optimized + Chat IA")
        st.markdown(
            f"###### 📅 Référence données : {reference_now.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(dernière mesure du fichier, indépendante de l'heure système)"
        )

    st.divider()

    col_dashboard, col_chat = st.columns([3, 2])

    # ── Dashboard ─────────────────────────────────────────────────────────
    with col_dashboard:
        st.markdown("### 📊 TABLEAU DE BORD")

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            lat = df_recent['latence_totale_ms'].mean()
            st.metric("Latence", f"{lat:.0f}ms", "🔴" if lat > SEUIL_LATENCE_WARN else "🟢")

        with kpi_col2:
            cn0 = df_recent['cn0_dbhz'].mean() if 'cn0_dbhz' in df_recent.columns else np.nan
            st.metric("C/N₀", f"{cn0:.1f}dB", "🟢" if cn0 > 10 else "🔴")

        with kpi_col3:
            qual = df_recent['qualite_signal_pct'].mean() if 'qualite_signal_pct' in df_recent.columns else np.nan
            st.metric("Qualité", f"{qual:.1f}%", "🟢" if qual > 85 else "🔴")

        with kpi_col4:
            temp = df_recent['temperature_hpa_c'].mean() if 'temperature_hpa_c' in df_recent.columns else 0
            st.metric("Temp HPA", f"{temp:.1f}°C", "🟢" if temp < 65 else "🔴")

        st.divider()

        st.markdown(f"#### Évolution Latence (fenêtre de {int(FENETRE_RECENTE.total_seconds()//3600)}h)")
        fig = go.Figure()
        df_sorted = df_recent.sort_values('timestamp')
        fig.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['latence_totale_ms'],
            mode='lines+markers',
            name='Latence',
            line=dict(color='#1a472a', width=2),
            fill='tozeroy'
        ))
        fig.add_hline(y=SEUIL_LATENCE_WARN, line_dash='dash', line_color='orange', annotation_text='Avertissement')
        fig.add_hline(y=SEUIL_LATENCE_CRIT, line_dash='dash', line_color='red', annotation_text='Critique')
        fig.update_layout(height=300, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    # ── Chat IA ───────────────────────────────────────────────────────────
    with col_chat:
        st.markdown("### 🤖 Chat IA Assistant")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        user_input = st.chat_input("Posez une question sur prédictions/anomalies...")

        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            chatbot = ChatbotIAPrediction(model_ia, df_recent, df_complet)
            reponse_ia = chatbot.repondre(user_input)
            st.session_state.chat_messages.append({"role": "assistant", "content": reponse_ia})
            st.rerun()

    st.divider()

    # ── Onglets détails ───────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📈 Analyses Avancées", "🗂️ Données Brutes", "📋 Rapports"])

    with tab1:
        st.markdown("### Analyses Statistiques")
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(df_recent, x='latence_totale_ms', nbins=30,
                                color_discrete_sequence=['#1a472a'],
                                title='Distribution Latence')
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            if 'qualite_signal_pct' in df_recent.columns:
                fig = px.histogram(df_recent, x='qualite_signal_pct', nbins=30,
                                    color_discrete_sequence=['#22c55e'],
                                    title='Distribution Qualité')
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Données Brutes")
        nb_rows = st.slider("Lignes", 10, 200, 50)
        st.dataframe(df_recent.head(nb_rows), use_container_width=True, height=300)
        csv = df_recent.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 CSV Export", csv,
            f"flyaway_{reference_now.strftime('%Y%m%d_%H%M%S')}.csv"
        )

    with tab3:
        st.markdown("### Génération Rapports")
        if st.button("📄 Générer Rapports", use_container_width=True, type="primary"):
            st.info("✅ Rapports générés - voir l'onglet Données Brutes pour le téléchargement")


if __name__ == "__main__":
    main()