import requests, yaml, textwrap, string, re

QIDS = [
  "Q243", #Eiffel Tower
  "Q12501", # Great wall of china
  "Q676203", # Machu Pichu 
  "Q9202", # Statue of Liberty
  "Q9141", # Taj Mahal
  "Q10285", #Colosseum
  "Q41225", # Big Ben

  # --- Americas ---
  "Q44440", # Golden Gate Bridge
  "Q172822", # Hoover Dam
  "Q7350", # Panama Canal
  "Q5859", # Chichén Itzá
  "Q79961", # Christ the Redeemer
  "Q83497", # Mount Rushmore
  "Q274120", # Niagara Falls
  "Q134883", # CN Tower
  "Q181172", # Tikal
  "Q36332", # Iguazu Falls

  # --- Europe ---
  "Q39671", # Stonehenge
  "Q131013", # Acropolis of Athens
  "Q42182", # Buckingham Palace
  "Q39054", # Leaning Tower of Pisa
  "Q2470217", # Sagrada Família
  "Q129846", # Saint Basil's Cathedral
  "Q20892", # Mont Saint-Michel
  "Q1208", # brandenburg Gate
  "Q12512", # St. Peter's Basilica
  "Q47476", # Alhambra
  "Q12506", # Hagia Sophia
  "Q83125", # Tower Bridge

  # --- Asia ---
  "Q5788", # Petra
  "Q43473", # Angkor Wat
  "Q12495", # Burj Khalifa
  "Q80290", # Forbidden City
  "Q548679", # Marina Bay Sands
  "Q39231", # Mount Fuji
  "Q47672", # Terracotta Army
  "Q83063", # Petronas Towers
  "Q130958", # Great Sphinx of Giza

  # --- Africa ---
  "Q37200", # Great Pyramid of Giza
  "Q43278", # Victoria Falls
  "Q7296", # Mount Kilimanjaro
  "Q207590", # Laliebela
  "Q213360", # Table Mountain

  # --- Oceania ---
  "Q45178", # Sydney Opera House
  "Q33910", # Uluru (Ayers Rock)
  "Q7343", # Great Barrier Reef
  "Q187197", # Milford Sound

  ]  # add as needed

sparql = f"""
SELECT ?item ?enLabel (SAMPLE(?enTitle) AS ?wiki) 
       (SAMPLE(?officialURL) AS ?url)
       (SAMPLE(?architectLabel) AS ?architect)
       (SAMPLE(?styleLabel) AS ?style)
       (SAMPLE(?inception) AS ?inception_date)
       (SAMPLE(?visitor_count) AS ?visitors)
       (SAMPLE(?unesco_id) AS ?unesco)
       (SAMPLE(?lp_id) AS ?lonely_planet)
       (SAMPLE(?lat) AS ?lat) (SAMPLE(?lon) AS ?lon)
       (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases)
WHERE {{
  VALUES ?item {{ {' '.join(f'wd:{q}' for q in QIDS)} }}
  ?item rdfs:label ?enLabel FILTER(LANG(?enLabel)="en").
  OPTIONAL {{ 
    ?item wdt:P84 ?architect. 
    ?architect rdfs:label 
    ?architectLabel FILTER(LANG(?architectLabel)="en"). 
  }}
  
  OPTIONAL {{ 
    ?item wdt:P149 ?style. 
    ?style rdfs:label ?styleLabel FILTER(LANG(?styleLabel)="en"). 
  }}
  
  OPTIONAL {{ ?item wdt:P571 ?inception. }}

  OPTIONAL {{ ?item wdt:P1174 ?visitor_count. }}

  OPTIONAL {{ ?item wdt:P757 ?unesco_id. }}

  OPTIONAL {{ ?item wdt:P2792 ?lp_id. }}

  OPTIONAL {{ ?item wdt:P856 ?officialURL . }}

  OPTIONAL {{
    ?article schema:about ?item ;
             schema:isPartOf <https://en.wikipedia.org/> ;
             schema:name ?enTitle .
  }}
  OPTIONAL {{
    ?item wdt:P625 ?c .
    BIND(geof:latitude(?c)  AS ?lat)
    BIND(geof:longitude(?c) AS ?lon)
  }}
  OPTIONAL {{
    ?item skos:altLabel ?alias FILTER(LANG(?alias)="en")
  }}
}}
GROUP BY ?item ?enLabel
"""

URL = "https://query.wikidata.org/sparql"
data = requests.get(URL, params={"format":"json", "query":sparql},
                    headers={"User-Agent":"LandmarkRAG/0.1"}).json()

def create_slug(text: str)-> str:
  """ convert string like Eiffel Tower to eiffel_tower """
  text = re.sub(r"[^\w\s]", '', text)
  text = re.sub(r"\s+", ' ', text)  # normalize whitespace
  return text.replace(" ","_").lower()

records=[]
for row in data["results"]["bindings"]:
    qid   = row["item"]["value"].split('/')[-1]      # Q243
    label = row["enLabel"]["value"]                  # Eiffel Tower
    title = row.get("wiki", {}).get("value","")      # Eiffel_Tower
    lat   = float(row["lat"]["value"]) if "lat" in row else None
    lon   = float(row["lon"]["value"]) if "lon" in row else None
    aliases = row["aliases"]["value"].split("|") if row.get("aliases") else []
    architect = row.get("architect", {}).get("value")
    style     = row.get("style", {}).get("value")
    inception = row.get("inception_date", {}).get("value")
    visitors  = row.get("visitors", {}).get("value")
    unesco   = row.get("unesco", {}).get("value")
    lonely_planet = row.get("lonely_planet", {}).get("value")
    # cheap slug for id (ET, SL, etc.) 
    slug = create_slug(label)

    records.append({
        "id": slug,
        "name": label,
        "wikidata_qid": qid,
        "wikipedia_title": title,
        "official_website_url": row.get("url", {}).get("value"),
        "architect": architect,
        "style": style,
        "inception_date": inception,
        "annual_visitors": visitors,
        "unesco_id": unesco,
        "lonely_planet_id": lonely_planet,
        "aliases": aliases[:5],      
        "location": {"lat":lat, "lon":lon}
    })

# sort by name
records = sorted(records, key=lambda r: r["name"])

yaml.safe_dump({"landmarks": records, "evaluation_questions":[]},
               open("manifests/landmarks.yaml","w"),
               sort_keys=False)
print("wrote manifests/landmarks.yaml")