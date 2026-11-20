from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import io
import json
from pydub import AudioSegment
from datetime import datetime

app = FastAPI(
    title="API AudiSense Mobile", 
    description="Back-end de détection de bruits domestiques pour malentendants"
)

historique_bruits = []

TRADUCTION_CLASSES = {
    "Crying Baby": {"nom": "Bébé qui pleure", "emoji": "👶🚨", "watch": "BÉBÉ PLEURE 👶"},
    "Siren": {"nom": "Sirène d'ambulance / Police", "emoji": "📢🚨", "watch": "SIRÈNE ALERTE 📢"},
    "Clapping": {"nom": "Applaudissements", "emoji": "👏🎉", "watch": "APPLAUDISSEMENTS 👏"},
    "Door Wood Knock": {"nom": "Quelqu'un frappe à la porte", "emoji": "🚪✊", "watch": "PORTE : TOC-TOC 🚪"},
    "Glass Breaking": {"nom": "Verre / Vaisselle cassée", "emoji": "💥🥛", "watch": "VERRE CASSÉ 💥"},
    "Car Horn": {"nom": "Klaxon de voiture", "emoji": "🚗🔊", "watch": "KLAXON PROCHE 🚗"},
    "Pouring Water": {"nom": "Eau qui coule / Fuite", "emoji": "💧🚰", "watch": "FUITE D'EAU 💧"},
    "Clock Alarm": {"nom": "Alarme / Réveil qui sonne", "emoji": "⏰🔔", "watch": "ALARME REVEIL ⏰"},
    "Washing Machine": {"nom": "Machine à laver", "emoji": "🧺🔔", "watch": "MACHINE À LAVER 🧺"},
    "Crackling Fire": {"nom": "Crépitement de feu / Danger", "emoji": "🔥⚠️", "watch": "FEU / DANGER 🔥"}
}

# --- RECONSTRUCTION DIRECTE DE L'ARCHITECTURE MULTI-COUCHE ROBUSTE ---
def bâtir_architecture_reseau(input_dim=1024, num_classes=10):
    model = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    return model

print("Chargement des modèles IA...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# 1. Instanciation de l'architecture propre
model_classification = bâtir_architecture_reseau(input_dim=1024, num_classes=10)

# 2. Chargement pur des poids entraînés (Exclut toutes les métadonnées conflictuelles)
try:
    model_classification.load_weights('modele_bruits_domestiques_robuste.keras')
    print("Poids du modèle robuste injectés avec succès.")
except Exception as e:
    print(f"Tentative alternative d'extraction des poids : {e}")
    # Sécurité si le fichier contient l'objet complet : charger via l'API interne brute
    _tmp_model = tf.keras.models.load_model('modele_bruits_domestiques_robuste.keras', compile=False)
    model_classification.set_weights(_tmp_model.get_weights())
    print("Poids du modèle extraits et synchronisés de force.")

with open('mapping_classes.json', 'r') as f:
    mapping = json.load(f)
index_vers_classe = {int(idx): classe for classe, idx in mapping.items()}
print("Modèles chargés avec succès.")

def traiter_flux_audio(audio_bytes):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio_segment.get_array_of_samples())
    
    if audio_segment.sample_width == 2:
        samples_float = samples.astype(np.float32) / 32768.0
    elif audio_segment.sample_width == 4:
        samples_float = samples.astype(np.float32) / 2147483648.0
    else:
        max_val = np.max(np.abs(samples))
        samples_float = samples.astype(np.float32) / max_val if max_val > 0 else samples.astype(np.float32)
        
    target_length = 16000 * 5
    if len(samples_float) < target_length:
        samples_float = np.pad(samples_float, (0, target_length - len(samples_float)), 'constant')
    else:
        samples_float = samples_float[:target_length]
    return samples_float

@app.post("/analyser-bruit/")
async def analyser_bruit(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        waveform = traiter_flux_audio(audio_bytes)
        
        max_amplitude = float(np.max(np.abs(waveform)))
        rms = float(np.sqrt(np.mean(waveform**2)))
        
        if rms < 0.005:
            return {"statut": "silence", "message": "Aucun son significatif détecté."}
            
        waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, embeddings, _ = yamnet_model(waveform_tensor)
        
        if embeddings.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Caractéristiques audio introuvables.")
            
        predictions_par_tranche = model_classification.predict(embeddings)
        probabilites = np.max(predictions_par_tranche, axis=0)
        
        id_classe_predite = int(np.argmax(probabilites))
        score_confiance = float(probabilites[id_classe_predite])
        nom_classe_en = index_vers_classe[id_classe_predite].replace('_', ' ').title()
        
        diagnostics_scores = {}
        for idx, prob in enumerate(probabilites):
            nom_en = index_vers_classe[idx].replace('_', ' ').title()
            trad_nom = TRADUCTION_CLASSES.get(nom_en, {"nom": nom_en})["nom"]
            diagnostics_scores[trad_nom] = f"{prob:.1%}"

        if score_confiance >= 0.5:
            infos_classe = TRADUCTION_CLASSES.get(nom_classe_en, {"nom": "Inconnu", "emoji": "🔔", "watch": "BRUIT"})
            heure_actuelle = datetime.now().strftime("%H:%M:%S")
            
            alerte_data = {
                "statut": "alerte_active",
                "nom_bruit_fr": infos_classe["nom"],
                "emoji": infos_classe["emoji"],
                "confiance": f"{score_confiance:.1%}",
                "duree_clignotant_secondes": 10,
                "notification_montre": infos_classe["watch"],
                "heure_detection": heure_actuelle
            }
            
            historique_bruits.append({"heure": heure_actuelle, "nom": infos_classe["nom"], "emoji": infos_classe["emoji"]})
            return {"alerte": alerte_data, "diagnostics": {"amplitude_max": max_amplitude, "rms": rms, "scores": diagnostics_scores}}
        else:
            return {"statut": "bruit_inconnu", "message": "Bruit capté mais incertain."}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historique/")
async def obtenir_historique():
    return {"historique": historique_bruits}

@app.get("/health")
async def health():
    return {"statut": "ok"}

@app.get("/")
async def accueil():
    return FileResponse("index.html")
