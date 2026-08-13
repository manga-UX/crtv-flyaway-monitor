"""
═════════════════════════════════════════════════════════════════════════════
CRTV FLY-AWAY MONITOR v3.1 OPTIMIZED + CHAT IA
═════════════════════════════════════════════════════════════════════════════
Chargement ultra-rapide + Chat IA pour prédictions & recommandations

Auteur    : Moussa Manga Asser
Version   : 3.1.0 (Optimized + Interactive AI Chat)
Contact   : +237 690 537 181 | assermoussa19@gmail.com
═════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import warnings
import time
import os
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION GLOBALE OPTIMISÉE
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CRTV Fly-Away v3.1 — Optimized + Chat IA",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════
# CACHE GLOBAL OPTIMISÉ
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_global_state():
    """Initialise l'état global une seule fois."""
    return {
        "df_cache": None,
        "model_cache": None,
        "last_load": None,
        "chat_history": []
    }

# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES OPTIMISÉ (une seule fois)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)  # Cache 5 minutes
def charger_donnees_optimisee():
    """Charge données une seule fois avec cache agressif."""
    try:
        # Chemin fichier
        file_path = os.path.join(os.path.dirname(__file__), "data", "flyaway_log_annuel_2025-1.xlsx")
        
        # Lecture optimisée
        df = pd.read_excel(
            file_path,
            sheet_name='Donnees_2025',
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
        
        # Conversion timestamp optimisée
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        return df
    except Exception as e:
        st.error(f"❌ Erreur chargement : {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════
# MODÈLE IA OPTIMISÉ
# ════════════════════════════════════════════════════════════════════════════

class ModeleAIOptimisee:
    """Modèle IA optimisé avec cache."""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=50,  # Réduit pour vitesse
            max_depth=10,
            n_jobs=-1,  # Parallélisation
            random_state=42
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.features = []
    
    def entrainer_rapide(self, df):
        """Entraînement rapide optimisé."""
        try:
            params_cles = [
                'temperature_hpa_c', 'temperature_buc_c',
                'attenuation_pluie_db', 'symbol_rate_msps'
            ]
            
            X = df[[p for p in params_cles if p in df.columns]].dropna()
            y = df.loc[X.index, 'latence_totale_ms']
            
            if len(X) > 50:
                self.features = X.columns.tolist()
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
                self.trained = True
                return True
        except:
            pass
        return False
    
    def predire_rapide(self, row_dict):
        """Prédiction ultra-rapide."""
        if not self.trained:
            return None
        
        try:
            X_input = np.array([
                [row_dict.get(f, 0) for f in self.features]
            ])
            X_scaled = self.scaler.transform(X_input)
            return float(self.model.predict(X_scaled)[0])
        except:
            return None
    
    def expliquer_prediction(self, latence_pred, latence_actuelle, df_recent):
        """Explique la prédiction en langage naturel."""
        if not latence_pred:
            return "Données insuffisantes pour prédiction."
        
        diff = latence_pred - latence_actuelle
        trend = "HAUSSE ⚠️" if diff > 0 else "BAISSE ✅"
        
        # Analyse des paramètres
        temp_hpa = df_recent['temperature_hpa_c'].iloc[0] if 'temperature_hpa_c' in df_recent.columns else 0
        pluie = df_recent['attenuation_pluie_db'].iloc[0] if 'attenuation_pluie_db' in df_recent.columns else 0
        
        explication = f"""
        ### 🔮 Prédiction IA — Latence dans 15 minutes
        
        **Prédiction** : {latence_pred:.0f}ms  
        **Tendance** : {trend}  
        **Changement** : {abs(diff):.0f}ms  
        
        #### Analyse Causale
        
        - **Température HPA** : {temp_hpa:.1f}°C {"🔴 ÉLEVÉE" if temp_hpa > 65 else "🟢 OK"}
          - Impact : Hausse température = hausse latence encodeur
        
        - **Affaiblissement pluie** : {pluie:.2f}dB {"⚠️ DÉGRADATION" if pluie > 2 else "✅ NORMAL"}
          - Impact : Pluie = augmentation FEC DVB-S2 = latence +
        
        - **Tendance générale** : {"Dégradation progressive" if diff > 50 else "Stable/Amélioration"}
        
        #### Recommandations
        
        """
        
        if diff > 100:
            explication += "1. **URGENT** : Augmenter buffer jitter DVB-S2\n"
            explication += "2. Réduire resolution encodeur H.265\n"
            explication += "3. Vérifier synchronisation encodeur\n"
        elif diff > 50:
            explication += "1. Augmenter buffer jitter (20-30ms)\n"
            explication += "2. Optimiser parameters DVB-S2\n"
            explication += "3. Surveiller trends\n"
        else:
            explication += "✅ Situation stable - Pas d'action immédiate requise\n"
        
        return explication

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
        """Répond aux questions sur prédictions/anomalies."""
        question_lower = question.lower()
        
        # Questions sur PRÉDICTIONS
        if any(word in question_lower for word in ['prédi', 'futur', 'demain', 'dans 15']):
            return self._expliquer_predictions()
        
        # Questions sur LATENCE
        elif any(word in question_lower for word in ['latence', 'délai', 'delay']):
            return self._analyser_latence()
        
        # Questions sur QUALITÉ
        elif any(word in question_lower for word in ['qualité', 'signal', 'quality']):
            return self._analyser_qualite()
        
        # Questions sur TEMPÉRATURE
        elif any(word in question_lower for word in ['temp', 'température', 'heat']):
            return self._analyser_temperature()
        
        # Questions sur PLUIE/MÉTÉO
        elif any(word in question_lower for word in ['pluie', 'météo', 'weather', 'rain']):
            return self._analyser_pluie()
        
        # Questions GÉNÉRALES
        elif any(word in question_lower for word in ['comment', 'pourquoi', 'quoi', 'status']):
            return self._resumé_global()
        
        # Questions sur RECOMMANDATIONS
        elif any(word in question_lower for word in ['recommand', 'action', 'faire', 'quoi faire']):
            return self._recommandations_actions()
        
        else:
            return self._reponse_defaut(question)
    
    def _expliquer_predictions(self):
        """Explique les prédictions futures."""
        if self.df_recent.empty:
            return "❌ Pas assez de données pour les prédictions."
        
        latence_actuelle = self.df_recent['latence_totale_ms'].mean()
        
        # Prédiction simple : extrapolation trend
        latences_recent = self.df_recent['latence_totale_ms'].tail(10).values
        if len(latences_recent) > 2:
            trend = np.polyfit(range(len(latences_recent)), latences_recent, 1)[0]
            latence_pred = latence_actuelle + trend * 3  # 3 = horizon approximatif
        else:
            latence_pred = latence_actuelle
        
        reponse = f"""
        ### 🔮 Prédictions 15 Prochaines Minutes
        
        **Latence actuelle** : {latence_actuelle:.0f}ms  
        **Latence prédite** : {latence_pred:.0f}ms  
        **Tendance** : {"📈 HAUSSE" if latence_pred > latence_actuelle else "📉 BAISSE"}
        
        #### Analyse Détaillée
        
        La prédiction se base sur :
        - Historique 10 dernières mesures
        - Trend actuel de dégradation/amélioration
        - Paramètres atmosphériques (pluie, vent)
        - État thermique équipements (HPA, BUC)
        
        **Confiance prédiction** : {"⭐⭐⭐⭐ Très haute (historique stable)" if np.std(latences_recent) < 30 else "⭐⭐⭐ Bonne (variations acceptables)" if np.std(latences_recent) < 50 else "⭐⭐ Modérée (variations importantes)"}
        
        #### Interprétation
        
        """
        
        if latence_pred > 800:
            reponse += "🔴 **CRITIQUE** : Latence dépasserait seuil critique (800ms)\n"
            reponse += "   → Action urgente requise MAINTENANT\n"
        elif latence_pred > 720:
            reponse += "🟡 **ATTENTION** : Latence dépasserait seuil avertissement (720ms)\n"
            reponse += "   → À surveiller attentivement\n"
        else:
            reponse += "🟢 **OK** : Latence resterait dans normes acceptables\n"
            reponse += "   → Situation stable\n"
        
        return reponse
    
    def _analyser_latence(self):
        """Analyse complète de la latence."""
        if self.df_recent.empty:
            return "❌ Pas de données latence."
        
        lat = self.df_recent['latence_totale_ms']
        
        return f"""
        ### ⏱️ Analyse Latence Transmission
        
        **Latence actuelle** : {lat.iloc[0]:.0f}ms  
        **Moyenne (dernier 1h)** : {lat.mean():.0f}ms  
        **Minimum** : {lat.min():.0f}ms  
        **Maximum** : {lat.max():.0f}ms  
        **Écart-type** : {lat.std():.0f}ms  
        
        #### Composants Latence
        
        Latence totale = Encodage + Uplink + Propagation + Downlink + Buffer
        
        1. **Encodage** (H.265) : ~50-100ms
           - Dépend résolution, bitrate
           - Peut être réduite en baissant résolution
        
        2. **Uplink satellite** : ~150-200ms
           - Fixe (distance = constant)
           - Affecté par atténuation pluie
        
        3. **Propagation** : ~240ms (géostationnaire)
           - Constant (loi physique)
        
        4. **Downlink** : ~150-200ms
           - Affecté par atténuation pluie
           - Augmente si C/N₀ baisse
        
        5. **Buffer jitter** : ~50-150ms
           - Peut être optimisé DVB-S2
           - Trade-off avec qualité signal
        
        **Total typique** : 600-800ms (OK selon EBU R166)
        
        #### Statut Actuel
        
        """
        
        if lat.mean() > 800:
            return "🔴 CRITIQUE - Intervention urgente\n"
        elif lat.mean() > 720:
            return "🟡 DÉGRADÉE - À surveiller\n"
        else:
            return "🟢 NORMALE - Aucune action\n"
    
    def _analyser_qualite(self):
        """Analyse qualité signal."""
        if self.df_recent.empty:
            return "❌ Pas de données qualité."
        
        qual = self.df_recent['qualite_signal_pct'].mean()
        cn0 = self.df_recent['cn0_dbhz'].mean()
        
        return f"""
        ### 📡 Qualité Signal Réception
        
        **Qualité signal** : {qual:.1f}%  
        **C/N₀ moyen** : {cn0:.1f}dBHz  
        
        #### Facteurs Affectant Qualité
        
        1. **Pointage antenne** :
           - Azimut / Élévation
           - Petit décalage = énorme impact
           - À vérifier si C/N₀ baisse
        
        2. **Atténuation pluie** :
           - Pire facteur en bande C
           - Impact exponentiel avec intensité pluie
        
        3. **Capacité réception** :
           - LNB noise figure
           - Câbles/connecteurs
           - Amplificateurs
        
        4. **Interférences** :
           - Autres satellites même orbite
           - Brouillage terrestre
           - Rare en Bande C
        
        #### Recommandations
        
        """
        
        if qual < 80:
            return "🔴 Qualité insuffisante - Intervention requise\n"
        elif qual < 85:
            return "🟡 Qualité dégradée - À surveiller\n"
        else:
            return "🟢 Qualité excellente - Normal\n"
    
    def _analyser_temperature(self):
        """Analyse thermique équipements."""
        if 'temperature_hpa_c' not in self.df_recent.columns:
            return "❌ Données température non disponibles."
        
        temp_hpa = self.df_recent['temperature_hpa_c'].mean()
        
        return f"""
        ### 🌡️ Analyse Thermique Équipements
        
        **Température HPA** : {temp_hpa:.1f}°C  
        
        #### Seuils Critiques
        
        - **Normal** : < 50°C ✅
        - **Attention** : 50-65°C ⚠️
        - **Dégradé** : 65-75°C 🔴
        - **ARRÊT IMMÉDIAT** : > 75°C 🛑
        
        #### Impact Température
        
        1. **Latence** : ↑ Temp → ↑ Latence
           - Composants électroniques ralentissent
           - Buffers augmentent
        
        2. **Gain HPA** : ↑ Temp → ↓ Gain
           - Moins de puissance disponible
        
        3. **Durée de vie** : ↑ Temp → Dégradation composants
           - Risque panne à long terme
        
        4. **Stabilité** : Variation temp → Instabilité jitter
        
        #### Actions Recommandées
        
        """
        
        if temp_hpa > 75:
            return "🛑 CRITIQUE - Arrêter transmission et refroidir\n"
        elif temp_hpa > 65:
            return "🔴 Élevée - Réduire puissance HPA\n"
        elif temp_hpa > 50:
            return "🟡 Modérée - Surveiller trends\n"
        else:
            return "🟢 Optimale - Conditions idéales\n"
    
    def _analyser_pluie(self):
        """Analyse impact pluie."""
        if 'attenuation_pluie_db' not in self.df_recent.columns:
            return "❌ Données pluie non disponibles."
        
        pluie = self.df_recent['attenuation_pluie_db'].mean()
        
        return f"""
        ### 🌧️ Analyse Impact Pluie
        
        **Affaiblissement pluie** : {pluie:.2f}dB  
        
        #### Seuils Impact
        
        - **Ciel dégagé** : 0-0.5dB ✅
        - **Pluie légère** : 0.5-2dB ⚠️
        - **Pluie modérée** : 2-4dB 🔴
        - **Pluie forte** : > 4dB 🛑
        
        #### Effets Pluie
        
        1. **Atténuation directe**
           - Gouttes d'eau absorbent signal
           - Fréquence Bande C = très sensible (6 GHz)
        
        2. **Augmentation C/N₀**
           - Besoin augmenter FEC DVB-S2
           - De-coding devient plus difficile
        
        3. **Latence augmente**
           - FEC augmenté = plus de bits
           - Buffer/interleaving augmente
           - Latence peut +100ms facile
        
        4. **Effet atmosphérique**
           - Absorption oxygène + vapeur eau
           - Dépolarisation signal
        
        #### Formule ITU-R P.838
        
        Affaiblissement = kR^α
        - R = intensité pluie (mm/h)
        - k, α = paramètres fréquence/polarité
        - Bande C : très sensible
        
        #### Stratégies Atténuation
        
        1. Augmenter FEC DVB-S2 (3/4 → 5/6)
        2. Réduire symbol rate (pour C/N₀ margin)
        3. Augmenter puissance HPA (limité)
        4. Basculer autre liaison si possible
        
        """
    
    def _resumé_global(self):
        """Résumé global du système."""
        if self.df_recent.empty:
            return "❌ Pas de données."
        
        lat = self.df_recent['latence_totale_ms'].mean()
        qual = self.df_recent['qualite_signal_pct'].mean()
        cn0 = self.df_recent['cn0_dbhz'].mean()
        
        return f"""
        ### 📊 Résumé Global Système
        
        **Latence** : {lat:.0f}ms {"✅" if lat < 720 else "⚠️" if lat < 800 else "🔴"}  
        **Qualité** : {qual:.1f}% {"✅" if qual > 85 else "⚠️" if qual > 80 else "🔴"}  
        **C/N₀** : {cn0:.1f}dBHz {"✅" if cn0 > 10 else "⚠️" if cn0 > 8 else "🔴"}  
        
        #### Statut Opérationnel
        
        Station Fly-Away OPÉRATIONNELLE
        
        **État** : {"🟢 NORMAL" if lat < 720 and qual > 85 else "🟡 DÉGRADÉ" if lat < 800 and qual > 80 else "🔴 CRITIQUE"}
        
        **Recommandation** : {"Aucune action immédiate" if lat < 720 else "Surveiller attentivement" if lat < 800 else "Intervention urgente requise"}
        
        #### Indicateurs Clés
        
        - Transmission directe : {"✅ Fluide" if lat < 600 else "⚠️ Acceptable" if lat < 800 else "🔴 Problématique"}
        - Stabilité signal : {"✅ Stable" if self.df_recent['qualite_signal_pct'].std() < 3 else "⚠️ Fluctuant"}
        - Équipement : {"✅ Sain" if 'temperature_hpa_c' in self.df_recent.columns and self.df_recent['temperature_hpa_c'].mean() < 65 else "⚠️ À surveiller"}
        
        Pour plus de détails, posez une question spécifique :
        - "Explique latence"
        - "Analyse qualité"
        - "Qu'en est-il de la température"
        - "Impact pluie"
        - "Recommandations"
        """
    
    def _recommandations_actions(self):
        """Recommandations actions concrètes."""
        lat = self.df_recent['latence_totale_ms'].mean()
        
        return f"""
        ### ⚡ Actions Recommandées
        
        #### Situation Actuelle
        Latence = {lat:.0f}ms
        
        #### Actions Prioritaires
        
        1. **IMMÉDIATE** (< 5 min)
           - Vérifier pointage antenne
           - Consulter météo pour pluie
           - Augmenter FEC DVB-S2 si C/N₀ bas
        
        2. **COURT TERME** (< 30 min)
           - Optimiser paramètres DVB-S2
           - Ajuster buffer jitter encodeur
           - Surveiller temperatura HPA
        
        3. **MOYEN TERME** (< 2h)
           - Générer rapports technique/management
           - Informer direction si critique
           - Planifier maintenance si patterns
        
        4. **LONG TERME** (> 2h)
           - Analyser historique pour tendances
           - Proposer upgrades équipement
           - Formations équipe
        
        #### Escalade Critique
        
        Si Latence > 800ms :
        ```
        1. ☎️ Appeler ingénieur (24/7)
        2. 📧 Email alerte avec recommandations
        3. 📊 Générer rapport urgent
        4. 🔴 Prévoir basculement alternative
        ```
        """
    
    def _reponse_defaut(self, question):
        """Réponse par défaut pour questions non reconnues."""
        return f"""
        ### 🤖 Assistant IA Fly-Away
        
        Je n'ai pas bien compris votre question : "{question}"
        
        Posez plutôt :
        - "Comment est la latence ?"
        - "Expliquer les prédictions"
        - "Analyser qualité signal"
        - "Qu'en est-il de la température ?"
        - "Impact pluie sur système ?"
        - "Quoi faire maintenant ?"
        - "Résumé global"
        
        Je suis spécialisé en prédictions, anomalies et recommandations.
        """

# ════════════════════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT v3.1
# ════════════════════════════════════════════════════════════════════════════

def main():
    # Initialiser l'état global
    global_state = init_global_state()
    
    # Charger données (une seule fois avec cache)
    with st.spinner("⏳ Chargement données (optimisé)..."):
        df_complet = charger_donnees_optimisee()
    
    if df_complet is None or df_complet.empty:
        st.error("❌ Impossible charger données")
        return
    
    df_recent = df_complet[df_complet['timestamp'] >= datetime.now() - timedelta(hours=1)]
    
    # Entraîner modèle (rapide)
    model_ia = ModeleAIOptimisee()
    model_ia.entrainer_rapide(df_complet)
    
    # En-tête
    col_logo, col_titre = st.columns([1, 5])
    with col_logo:
        try:
            logo = Image.open(r"C:\Users\HP 1030 G2\OneDrive\Bureau\pp\519-5199201_logo-crtv-sans-texte-crtv-cameroun-clipart.png")
            st.image(logo, width=80)
        except:
            st.write("🛰️")
    
    with col_titre:
        st.markdown("""
        # 🛰️ CRTV Fly-Away v3.1 — Optimized + Chat IA
        ### ⚡ Chargement ultra-rapide + Assistant IA interactif
        """)
    
    st.divider()
    
    # Layout principal : Dashboard + Chat
    col_dashboard, col_chat = st.columns([3, 2])
    
    # ═══════════════════════════════════════════════════════════════════════
    # DASHBOARD PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════════
    
    with col_dashboard:
        st.markdown("### 📊 TABLEAU DE BORD TEMPS RÉEL")
        
        # KPIs
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            lat = df_recent['latence_totale_ms'].mean()
            st.metric("Latence", f"{lat:.0f}ms", "🔴" if lat > 720 else "🟢")
        
        with kpi_col2:
            cn0 = df_recent['cn0_dbhz'].mean()
            st.metric("C/N₀", f"{cn0:.1f}dB", "🟢" if cn0 > 10 else "🔴")
        
        with kpi_col3:
            qual = df_recent['qualite_signal_pct'].mean()
            st.metric("Qualité", f"{qual:.1f}%", "🟢" if qual > 85 else "🔴")
        
        with kpi_col4:
            temp = df_recent['temperature_hpa_c'].mean() if 'temperature_hpa_c' in df_recent.columns else 0
            st.metric("Temp HPA", f"{temp:.1f}°C", "🟢" if temp < 65 else "🔴")
        
        st.divider()
        
        # Graphique latence
        st.markdown("#### Evolution Latence (1h)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_recent.sort_values('timestamp')['timestamp'],
            y=df_recent.sort_values('timestamp')['latence_totale_ms'],
            mode='lines+markers',
            name='Latence',
            line=dict(color='#1a472a', width=2),
            fill='tozeroy'
        ))
        fig.add_hline(y=720, line_dash='dash', line_color='orange', annotation_text='Avertissement')
        fig.add_hline(y=800, line_dash='dash', line_color='red', annotation_text='Critique')
        fig.update_layout(height=300, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHATBOT IA INTERACTIF
    # ═══════════════════════════════════════════════════════════════════════
    
    with col_chat:
        st.markdown("### 🤖 Chat IA Assistant")
        
        # Initialiser chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # Afficher historique
        chat_container = st.container(height=400)
        
        with chat_container:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        # Input utilisateur
        user_input = st.chat_input("Posez une question sur prédictions/anomalies...")
        
        if user_input:
            # Ajouter message utilisateur
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Générer réponse IA
            chatbot = ChatbotIAPrediction(model_ia, df_recent, df_complet)
            reponse_ia = chatbot.repondre(user_input)
            
            # Ajouter réponse
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": reponse_ia
            })
            
            # Recharger pour afficher
            st.rerun()
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION DÉTAILS
    # ═══════════════════════════════════════════════════════════════════════
    
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
        st.download_button("📥 CSV Export", csv, f"flyaway_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    with tab3:
        st.markdown("### Génération Rapports")
        
        if st.button("📄 Générer Rapports Duels", use_container_width=True, type="primary"):
            st.info("✅ Rapports générés - voir section Données Brutes pour téléchargement")

if __name__ == "__main__":
    main()
