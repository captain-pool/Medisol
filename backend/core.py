from flask import Flask, jsonify, request
from flask_cors import CORS
import PIL
import keras.backend as K
from datetime import datetime
import googlemaps
import pickle
import numpy as np
import uuid
import os
import tempfile
from keras.models import load_model
app = Flask(__name__)
CORS(app)
DICT={'mal': 'Malaria', 'pnm': 'Pnemonia', 'cancer': 'Breast Cancer'}
with open("API-Key", "r") as f:
  v = f.read()

client = googlemaps.Client(key=v)
current_location = googlemaps.geolocation.geolocate(client)
def query_map(key):
  map_ = googlemaps.places.places(client, "%s Hospitals"%DICT[key], location=current_location["location"])
  places = []
  for i in map_["results"]:
    places.append({"name": i["name"], "address": i["formatted_address"], "location": i["geometry"]["location"]})
  return places[:2]
@app.route("/", methods = ["POST", "GET"])
def index():
  
  if request.method == "POST":
    type_ = request.form.get("type", None)
    print(type_)
    data = None
    if 'img' in request.files:
      file_ = request.files['img']
      name = os.path.join(tempfile.gettempdir(), str(uuid.uuid4().hex[:10]))
      file_.save(name)
      print("[DEBUG: %s]"%datetime.now(),name)
      data = np.asarray(PIL.Image.open(name).resize([64, 64]), dtype=np.float32)
      data = np.resize(data, (1,64,64,3))
    final_json = []
    if not type_ is None and type_ == "cancer":
      fields = list(map(lambda x: "%s_mean"%x, ["texture", "perimeter", "smoothness", "compactness", "symmetry"]))
      data = np.asarray([request.args.get(x, 0) for x in fields], dtype=np.float32)
    for model in get_model(type_):
      preds, pred_val = translate(model["model"].predict(data), model["type"])
      places = []
      if preds != "None" and pred_val>0:
        places = query_map(model["type"])
      final_json.append({"empty": False, "type":model["type"], "prediction":preds, "pred_val": pred_val, "places": places})
    K.clear_session()
    return jsonify(final_json)
  return jsonify({"empty":True})
def load_model_(model_name):
  model_name = os.path.join("weights",model_name)
  print(model_name)
  if model_name.split(".")[-1].lower() == "h5":
    m = load_model(model_name)
    print(m)
  if model_name.split(".")[-1].lower() == "sav":
    with open(model_name, "rb") as f:
      m = pickle.load(f)
  return m
def translate(preds, type_):
  print(preds)
  if type_ == "pnm":
    if preds[0]==0:
      return "Normal",0
    else:
      return "Pneumonia",1
  elif type_=="mal":
    if preds[0]==0:
      return "Parasites",1
    else:
      return "Uninfected",0
  elif type_=="cancer":
    if preds[0]==0:
      return "Benign",0
    else:
      return "Malignant,",1
  return "None"
    
def get_model(name = None):
  l = []
  if name is None:
    l = [{"model": load_model_(x), "type": x.split(".")[-1]} for x in ["mal.h5", "pnm.h5", "cancer.sav"]]
  elif name == "pnm":
    l.append({"model": load_model_("pnm.h5"), "type": "pnm"})
  elif name == "mal":
    l.append({"model": load_model_("mal.h5"), "type": "mal"})
  elif name == "brc":
    l.append({"model": load_model_("cancer.sav"), "type": "cancer"})
  return l
app.run("0.0.0.0",8000, debug = False)
