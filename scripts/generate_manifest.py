import requests, yaml, textwrap, string

QIDS = ["Q243", "Q12501", "Q676203", "Q9202", "Q9141", "Q10285"]  # add as needed

sparql = f"""
SELECT ?item ?enLabel (SAMPLE(?enTitle) AS ?wiki) 
       (SAMPLE(?lat) AS ?lat) (SAMPLE(?lon) AS ?lon)
       (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases)
WHERE {{
  VALUES ?item {{ {' '.join(f'wd:{q}' for q in QIDS)} }}
  ?item rdfs:label ?enLabel FILTER(LANG(?enLabel)="en").
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

records=[]
for row in data["results"]["bindings"]:
    qid   = row["item"]["value"].split('/')[-1]      # Q243
    label = row["enLabel"]["value"]                  # Eiffel Tower
    title = row.get("wiki", {}).get("value","")      # Eiffel_Tower
    lat   = float(row["lat"]["value"]) if "lat" in row else None
    lon   = float(row["lon"]["value"]) if "lon" in row else None
    aliases = row["aliases"]["value"].split("|") if row.get("aliases") else []

    # cheap slug for id (ET, SL, etc.) – tweak as you like
    slug = "".join(ch for ch in label.title() if ch in string.ascii_letters)[:2]

    records.append({
        "id": slug,
        "name": label,
        "wikidata_qid": qid,
        "wikipedia_title": title,
        "aliases": aliases[:5],      # keep it short for now
        "location": {"lat":lat, "lon":lon}
    })

yaml.safe_dump({"landmarks": records, "evaluation_questions":[]},
               open("manifests/landmarks.yaml","w"),
               sort_keys=False)
print("wrote manifests/landmarks.yaml")