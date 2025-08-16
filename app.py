import os, json, time, pickle, numpy as np
import logging
import re
from flask import Flask, request, jsonify, render_template
from openai import OpenAI, OpenAIError, RateLimitError, AuthenticationError

# geopy imports
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# ── setup ─────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 5
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.2"))
EMBED_CACHE = "embeddings.pkl"
LOCATIONS_EMBED_CACHE = "locations_embeddings.pkl"
MAX_HISTORY = 10
MAX_DISTANCE_MILES = float(os.getenv("MAX_DISTANCE_MILES", "200"))

logging.basicConfig(level=logging.INFO, filename="chatbot.log", format="%(asctime)s - %(levelname)s - %(message)s")

if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY environment variable not set")
    print("Error: OPENAI_API_KEY environment variable not set")
    exit(1)

# initialize a geopy geolocator (with timeout)
geolocator = Nominatim(user_agent="eden_foods_chatbot (support@edenfoods.example)", timeout=5)

# ── load products ─────────────────────────────
try:
    with open("products.json") as f:
        products = json.load(f)
except (json.JSONDecodeError, FileNotFoundError) as e:
    logging.error(f"Error loading products.json: {e}")
    exit(1)

# ── load locations ────────────────────────────
try:
    with open("locations.json") as f:
        locations = json.load(f)
except (json.JSONDecodeError, FileNotFoundError) as e:
    logging.error(f"Error loading locations.json: {e}")
    print(f"Error loading locations.json: {e}")
    exit(1)

# ── embedding utility ─────────────────────────
def embed(text: str, retries=3, backoff_factor=2) -> np.ndarray:
    for attempt in range(retries):
        try:
            vec = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
            arr = np.array(vec, dtype=np.float32)
            denom = float(np.linalg.norm(arr))
            if denom < 1e-12:
                return arr
            return arr / denom
        except RateLimitError:
            if attempt == retries - 1:
                raise
            time.sleep(backoff_factor ** attempt)
        except AuthenticationError:
            logging.error("Invalid OpenAI API key")
            raise
        except OpenAIError as e:
            logging.warning(f"OpenAI error on attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(backoff_factor ** attempt)
    raise Exception("Failed to embed text after retries")

# ── build or load cached product vectors ──────
vecs = []  # ensure defined on first run
if os.path.exists(EMBED_CACHE):
    try:
        with open(EMBED_CACHE, "rb") as f:
            vecs = pickle.load(f)
        if len(vecs) != len(products):
            logging.warning("Embedding count mismatch. Regenerating embeddings...")
            vecs = []
    except Exception as e:
        logging.warning(f"Error loading embeddings.pkl: {e}. Regenerating...")
        vecs = []

if not vecs:
    vecs = []
    for i, p in enumerate(products, 1):
        text = " ".join([p["name"], p.get("description", ""), " ".join(p.get("tags", []))])
        try:
            vecs.append(embed(text))
        except Exception as e:
            logging.error(f"Failed to embed product {p['name']}: {e}")
            continue
        if i % 50 == 0 or i == len(products):
            print(f"embedded {i}/{len(products)}")
    with open(EMBED_CACHE, "wb") as f:
        pickle.dump(vecs, f)

for p, v in zip(products, vecs):
    p["vec"] = v

# ── build or load cached location vectors ─────
location_vecs = []
if os.path.exists(LOCATIONS_EMBED_CACHE):
    try:
        with open(LOCATIONS_EMBED_CACHE, "rb") as f:
            location_vecs = pickle.load(f)
        if len(location_vecs) != len(locations):
            logging.warning("Location embedding count mismatch. Regenerating embeddings...")
            location_vecs = []
    except Exception as e:
        logging.warning(f"Error loading locations_embeddings.pkl: {e}. Regenerating...")
        location_vecs = []

if not location_vecs:
    location_vecs = []
    for i, loc in enumerate(locations, 1):
        text = " ".join([loc["name"], loc["address"], " ".join(loc.get("promotions", []))])
        try:
            location_vecs.append(embed(text))
        except Exception as e:
            logging.error(f"Failed to embed location {loc['name']}: {e}")
            continue
        if i % 10 == 0 or i == len(locations):
            print(f"embedded {i}/{len(locations)} locations")
    with open(LOCATIONS_EMBED_CACHE, "wb") as f:
        pickle.dump(location_vecs, f)

for loc, v in zip(locations, location_vecs):
    loc["vec"] = v

# geocoding helper
def geocode_location(query: str):
    try:
        if not query:
            return None
        location = geolocator.geocode(query)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        logging.warning(f"Geocoding failed for '{query}': {e}")
    return None

# ---- small NLP helpers for intent ----
NEAR_ME_PATTERNS = re.compile(
    r"\b(near me|nearest|nearby|closest|close by|around me|"
    r"locations?|store|stores|"
    r"hours(?:\s+in|\s+near)?|hours|open now|"
    r"promotion|promotions?|deal|deals|special|specials|sale|sales)\b",
    re.I
)

def looks_near_me(q: str) -> bool:
    q = (q or "").strip().lower()
    if not q:
        return True
    return bool(NEAR_ME_PATTERNS.search(q))

# NEW: compute an origin label for the distance line
def compute_origin_label(query: str, lat: float, lon: float) -> str:
    q = (query or "").strip()
    if looks_near_me(q) and lat is not None and lon is not None:
        return "you"
    if q:
        # strip common prefixes like "hours in/near", "stores in/near"
        s = re.sub(r"^\s*(hours?|locations?|stores?)\s+(?:in|near)\s+", "", q, flags=re.I).strip()
        s = re.sub(r"^\s*hours(?:\s+(in|near))?\s*", "", s, flags=re.I).strip() or q
        return s
    return ""

# ── semantic search for products ──────────────
def search_products(query: str, limit: int = TOP_K):
    q_vec = embed(query)
    scored = [(float(q_vec @ p["vec"]), p) for p in products if "vec" in p]
    relevant = [(score, p) for score, p in scored if score > SCORE_THRESHOLD]
    relevant.sort(key=lambda x: -x[0])
    return [p for _, p in relevant[:limit]]

# ── nearest search for locations with radius expansion + text fallback ──
def search_locations(query: str, limit: int = TOP_K, lat: float = None, lon: float = None):
    query_norm = (query or "").strip()
    # strip leading "hours in/near" if user typed that
    query_clean = re.sub(r"^\s*hours(?:\s+(in|near))?\s*", "", query_norm, flags=re.I)

    # Decide coords: use browser coords for near-me style, otherwise geocode the typed place
    coords = None
    if looks_near_me(query_norm):
        if lat is not None and lon is not None:
            coords = (lat, lon)
        else:
            coords = geocode_location(query_clean)
    else:
        coords = geocode_location(query_clean)

    def within_radius(center, radius_miles):
        distances = []
        for loc in locations:
            if "latitude" in loc and "longitude" in loc:
                miles = geodesic(center, (loc["latitude"], loc["longitude"])).miles
                if miles <= radius_miles:
                    distances.append((miles, loc))
        distances.sort(key=lambda x: x[0])
        top = []
        for miles, loc in distances[:limit]:
            loc_with_distance = dict(loc)
            loc_with_distance["distance_miles"] = round(miles, 1)
            logging.info(f"  - Distance: {loc['name']}: {miles:.2f} miles (r={radius_miles})")
            top.append(loc_with_distance)
        return top

    # If we have coords, try progressive radius expansion (handles big states like Alaska)
    if coords:
        logging.info(f"search_locations using coords {coords} (query='{query_norm}')")
        for radius in (MAX_DISTANCE_MILES, 400, 800):
            top = within_radius(coords, radius)
            if top:
                return top
        logging.info(f"No stores within expanded radii of {coords}")

    # Fallback: if coords failed or radius expansion found nothing, do a light text match
    ql = query_norm.lower()
    text_hits = []
    for loc in locations:
        hay = f"{loc.get('name','')} {loc.get('address','')}".lower()
        if ql and ql in hay:
            text_hits.append(loc)

    if text_hits:
        return text_hits[:limit]

    # Nothing matched
    return []

# ── store info utility (promotions excluded) ──────────────
def get_store_info(info_type: str, query: str = None, lat: float = None, lon: float = None):
    info = {
        "return_policy": "Returns accepted within 30 days with receipt."
    }
    try:
        if info_type == "locations":
            origin = compute_origin_label(query or "", lat, lon)

            # If no query but we have user coords, use them to find nearest stores
            if not query and lat is not None and lon is not None:
                results = search_locations("", limit=TOP_K, lat=lat, lon=lon)
                if not results:
                    return f"No stores found matching your location within {MAX_DISTANCE_MILES} miles."
                lines = []
                for i, loc in enumerate(results, 1):
                    block = f"{i}. {loc['name']}\n"
                    if "distance_miles" in loc:
                        block += f"🚗 {loc['distance_miles']} miles from you\n"
                    block += (
                        f"📞 {loc.get('phone', 'Not available')}\n"
                        f"🕒 {loc.get('hours', 'Not available')}\n"
                        f"📍 {loc['address']}"
                    )
                    lines.append(block)
                return "\n\n".join(lines)

            if not query:
                # legacy: show a few stores when no location info at all (no distance available)
                with open("locations.json") as f:
                    locations_local = json.load(f)
                return "\n".join([
                    f"{i}. {loc['name']}\n"
                    f"📞 {loc.get('phone', 'Not available')}\n"
                    f"🕒 {loc.get('hours', 'Not available')}\n"
                    f"📍 {loc['address']}"
                    for i, loc in enumerate(locations_local[:TOP_K], 1)
                ]) + "\n\nMore locations available. You can allow location sharing or specify a city or ZIP code."

            # Query provided → use coords only if "near me"; otherwise geocode the place
            results = search_locations(query, limit=TOP_K, lat=lat, lon=lon)
            if not results:
                return f"No stores found matching your query within {MAX_DISTANCE_MILES} miles. You can allow location sharing or specify a different city or ZIP code."
            origin_label = compute_origin_label(query or "", lat, lon)
            lines = []
            for i, loc in enumerate(results, 1):
                block = f"{i}. {loc['name']}\n"
                if "distance_miles" in loc:
                    label = "you" if origin_label == "you" else origin_label
                    block += f"🚗 {loc['distance_miles']} miles from {label}\n"
                block += (
                    f"📞 {loc.get('phone', 'Not available')}\n"
                    f"🕒 {loc.get('hours', 'Not available')}\n"
                    f"📍 {loc['address']}"
                )
                lines.append(block)
            return "\n\n".join(lines)

        elif info_type == "hours":
            # Use coords when available, even without a query
            target = query or ""
            results = search_locations(target, limit=TOP_K, lat=lat, lon=lon)
            if not results:
                if (not target or re.fullmatch(r"hours?", target, re.I)) and (lat is None or lon is None):
                    return ("I can show hours near you, but I’ll need a city/ZIP or your location. "
                            "Please enable location sharing or say something like 'hours in Seattle'.")
                return f"No stores found matching your query within {MAX_DISTANCE_MILES} miles."
            # Numbered output: ONLY name + hours (two lines per store)
            return "\n\n".join([f"{i}. {loc['name']}\n🕒 {loc['hours']}" for i, loc in enumerate(results, 1)])
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.warning(f"Error loading locations.json: {e}")
    return info.get(info_type, "Unknown information requested.")

# ── tool schema ───────────────────────────────
functions = [
    {
        "name": "search_products",
        "description": "Semantic search over Eden Foods product catalog",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_locations",
        "description": "Find nearest stores; uses browser coords only for 'near me/locations' style queries or empty query, otherwise geocodes the typed place. Returns `distance_miles`, capped by MAX_DISTANCE_MILES.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_store_info",
        "description": "Retrieve general store information or specific details based on a query",
        "parameters": {
            "type": "object",
            "properties": {
                "info_type": {"type": "string", "enum": ["hours", "locations", "return_policy"]},
                "query": {"type": "string", "description": "Optional query for specific locations or hours (e.g., 'Anchorage' or 'near me')"}
            },
            "required": ["info_type"]
        }
    }
]

# ── GPT setup (prompt tuned: numbering + promo emoji-first + no emoji map) ─────────────
system_prompt = (
    "You are Eve, a kind, sweet, gentle, and helpful assistant for Eden Foods that uses many emojis to express your mood and emphasize the topic. However, never use heart emojis. "
    "Use many emojis when talking about products, store hours, store locations, return policy, and promotions—at least one emoji per line, ideally several. "
    "Be consistent and enthusiastic in all such responses unless the user expresses a negative emotion. "
    "If the user expresses sadness, anger, frustration, or uses phrases like 'I hate you' or 'I'm upset,' then respond with empathy and avoid using any emojis at all. "
    "For all other Eden Foods-related queries, always use a very cheerful tone and fill your response with joyful, thematic emojis (e.g., 🥬🍫🕒📍🛒🌟✨). "
    "Avoid any map-style emoji lines; do not show an emoji map. "
    "When the shopper asks for a product, use `search_products` to find up to 5 products that best match the query based on semantic similarity. "
    "Include only the name, price, aisle, and description in the response. Strictly do not include more than five products, even if more are available. "
    "Return fewer than 5 if there are not enough relevant matches. If no products are found, suggest related categories (e.g., dairy, poultry, sauces, international foods, candy) or ask the user to rephrase. "
    "For store hours, locations, or policies, use `get_store_info`. "
    "For specific location queries (e.g., 'stores in Anchorage'), use `get_store_info` with `info_type='locations'` to return name, phone, hours, and address. "
    "When listing locations, number each store (1., 2., 3.) and, if a distance is provided, render it exactly as '🚗 <miles> miles from you' or '🚗 <miles> miles from <place>'. Use the word 'miles', not 'mi'. "
    "For promotion queries (e.g., 'coffee promotions'), use `search_locations` to find up to 5 stores with relevant promotions, returning only the store name and promotions. If no stores match, suggest rephrasing the query. "
    "When listing promotions, number each store (1., 2., 3.) and show each promotion on its own line with an emoji at the start of the line (e.g., '🥑 10% off avocados'). "
    "When returning locations from any source, do **not** include extra labels like 'Address:', 'Phone:', 'Hours:', or 'Distance:'. "
    "Format each store block in this exact order with icons only: "
    "1) Store name on its own line; "
    "2) If available, show distance as '🚗 <miles> miles' and, if the location object includes an 'origin' field, render it as '🚗 <miles> miles from <origin>' where origin is either 'you' or the place string; "
    "3) A line with '📞 <phone>'; "
    "4) A line with '🕒 <hours>'; "
    "5) A line with '📍 <full address>'. "
    "Never invent items or information—only use provided functions. Use double newlines between store blocks for clarity. "
    "For hours responses specifically, return only the store name and the hours line, and number each store. "
)

history = [{"role": "system", "content": system_prompt}]

# ── Flask setup ───────────────────────────────
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global history

    data = request.json
    user_input = data.get("message", "").strip()

    # optional coordinates from the browser
    lat = data.get("lat", None)
    lon = data.get("lon", None)
    try:
        lat = float(lat) if lat is not None else None
        lon = float(lon) if lon is not None else None
    except (TypeError, ValueError):
        lat, lon = None, None

    if not user_input:
        return jsonify({"response": "Please enter a valid query."}), 400
    if len(user_input) > 500:
        return jsonify({"response": "Query is too long. Please shorten it."}), 400

    logging.info(f"User query: {user_input} (lat={lat}, lon={lon})")

    # --- EARLY: detect promotions from ORIGINAL text and handle directly
    orig_lower = user_input.lower()
    promo_intent = bool(re.search(r"\b(promotions?|deals?|specials?|sale|sales)\b", orig_lower))

    # Extract optional promo keyword (e.g., "chocolate promotions", "meat deals")
    promo_keyword = None
    m_kw = re.search(r"^(.*?)(?:\s+)?(?:promotions?|deals?|specials?|sale|sales)\b", user_input, re.I)
    if m_kw:
        candidate = (m_kw.group(1) or "").strip()
        if candidate and not re.fullmatch(r"(in|near)", candidate, re.I):
            promo_keyword = candidate.lower()

    # Extract an optional place (e.g., "in Anchorage", "near Orlando")
    place = None
    m_place = re.search(r"\b(?:in|near)\s+(.+)$", user_input, re.I)
    if m_place:
        place = m_place.group(1).strip()

    if promo_intent:
        query_for_geo = place or user_input
        results = search_locations(query_for_geo, limit=TOP_K, lat=lat, lon=lon)

        if not results:
            if (lat is None or lon is None) and not place:
                return jsonify({"response": (
                    "I can find promotions near you, but I’ll need your location or a city/ZIP. "
                    "Please enable location sharing or try ‘promotions in Orlando’."
                )}), 200
            return jsonify({"response": f"No stores found matching your query within {MAX_DISTANCE_MILES} miles."}), 200

        # Filter promos by keyword if provided
        cleaned = []
        for loc in results:
            promos = loc.get("promotions", [])
            if promo_keyword:
                keeps = [p for p in promos if promo_keyword in p.lower()]
                if keeps:
                    cleaned.append({"name": loc["name"], "promotions": keeps})
            else:
                cleaned.append({"name": loc["name"], "promotions": promos or ["None"]})

        # Fallback if keyword filters out everything
        if promo_keyword and not cleaned:
            cleaned = [{"name": loc["name"], "promotions": loc.get("promotions", ["None"])} for loc in results]

        # Let the model render the nice numbered/emoji-first promo response
        history.append({"role": "user", "content": user_input})
        history.append({"role": "function", "name": "search_locations", "content": json.dumps(cleaned[:TOP_K])})

        final_msg = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history
        ).choices[0].message.content.strip()

        history.append({"role": "assistant", "content": final_msg})
        history = [history[0]] + history[-(MAX_HISTORY-1):]
        logging.info(f"Assistant response (promos): {final_msg}")
        return jsonify({"response": final_msg}), 200

    # --- Otherwise, continue with rewrite + function-calling
    rewrite_prompt = (
        "You are a helpful assistant that reformulates queries to clarify intent for product or location searches while preserving the original query for non-product/non-location inputs, especially those with emotional content. "
        "For product queries, ensure specificity: 'sour cream' should include only sour cream or sour cream-flavored products like dips or chips, not unrelated items like sourdough pretzels or pickles. 'Sour treats' should include sour candies, not dairy. 'Thai' should include Thai-specific ingredients like curry paste, not generic seafood unless Thai-related. 'Spicy chicken' should include spicy chicken products or marinades, not generic spicy foods like salsa unless chicken-related. "
        "For location queries, preserve city or store name (e.g., 'stores in Orlando' → 'Orlando', 'Eden Foods Los Angeles Downtown' → 'Eden Foods Los Angeles Downtown'). "
        "For promotion queries, keep the promotion type (e.g., 'coffee promotions' → 'coffee promotions'). "
        "For queries that are not clearly about products, locations, or promotions (e.g., 'I hate you,' 'I'm upset,' or general statements), return the original query verbatim to preserve emotional tone and intent. "
        "Return a concise query that captures the core intent, avoiding overgeneralization."
    )
    try:
        rewrite = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": rewrite_prompt},
                {"role": "user", "content": user_input}
            ]
        ).choices[0].message.content.strip()
        logging.info(f"Rewritten query: {rewrite}")
    except Exception as e:
        logging.error(f"Query rewrite failed: {e}")
        return jsonify({"response": f"Query rewrite failed: {e}"}), 500

    history.append({"role": "user", "content": rewrite})

    first = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        functions=functions,
        function_call="auto"
    ).choices[0]

    # Handle function calls for both legacy .function_call and newer .tool_calls
    tool_name = None
    tool_args = None

    msg = first.message
    if getattr(msg, "function_call", None):
        tool_name = msg.function_call.name
        tool_args = json.loads(msg.function_call.arguments or "{}")
    elif getattr(msg, "tool_calls", None):
        call = msg.tool_calls[0]
        tool_name = call.function.name
        tool_args = json.loads(call.function.arguments or "{}")

    if tool_name:
        if tool_name == "search_products":
            if "limit" not in tool_args:
                tool_args["limit"] = TOP_K
            results = search_products(**tool_args)
            logging.info(f"Number of results from search_products: {len(results)}")
            if not results:
                history.append({
                    "role": "function",
                    "name": "search_products",
                    "content": "No products found. Suggest related categories or ask the user to rephrase their query."
                })
            else:
                limited_results = results[:TOP_K]
                cleaned = [
                    {"name": p["name"], "price": p["price"], "aisle": p["aisle"], "description": p["description"]}
                    for p in limited_results
                ]
                logging.info(f"Raw search results: {[p['name'] for p in results]}")
                history.append({
                    "role": "function",
                    "name": "search_products",
                    "content": json.dumps(cleaned)
                })

        elif tool_name == "search_locations":
            if "limit" not in tool_args:
                tool_args["limit"] = TOP_K
            results = search_locations(**tool_args, lat=lat, lon=lon)
            logging.info(f"Number of results from search_locations: {len(results)}")
            if not results:
                history.append({
                    "role": "function",
                    "name": "search_locations",
                    "content": "No stores found within 200 miles."
                })
            else:
                lower = rewrite.lower()
                is_promotion_query = "promotion" in lower
                promo_term = None
                m = re.search(r"(.+?)\s+promotions?", rewrite, re.I)
                if m:
                    promo_term = m.group(1).strip().lower()

                if is_promotion_query:
                    filtered = []
                    for loc in results:
                        promos = loc.get("promotions", [])
                        if promo_term:
                            keeps = [p for p in promos if promo_term in p.lower()]
                            if keeps:
                                filtered.append({"name": loc["name"], "promotions": keeps})
                        else:
                            filtered.append({"name": loc["name"], "promotions": promos or ["None"]})
                    if not filtered and results:
                        filtered = [{"name": loc["name"], "promotions": loc.get("promotions", ["None"])} for loc in results]
                    cleaned = filtered
                else:
                    # Compute origin label for distance line
                    origin_label = compute_origin_label(rewrite, lat, lon)
                    cleaned = []
                    for loc in results:
                        item = {
                            "name": loc["name"],
                            "address": loc["address"],
                            "phone": loc.get("phone", "Not available"),
                            "hours": loc.get("hours", "Not available"),
                        }
                        if "distance_miles" in loc:
                            item["distance_miles"] = loc["distance_miles"]
                            # pass origin for the prompt to render: "miles from you/place"
                            item["origin"] = "you" if origin_label == "you" else origin_label
                        cleaned.append(item)

                logging.info(f"Raw location results: {[loc['name'] for loc in results]}")
                history.append({
                    "role": "function",
                    "name": "search_locations",
                    "content": json.dumps(cleaned)
                })

        elif tool_name == "get_store_info":
            query = tool_args.get("query", None)
            result = get_store_info(tool_args["info_type"], query, lat=lat, lon=lon)
            history.append({"role": "function", "name": "get_store_info", "content": result})

        # Second turn to render the function result
        second = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history
        ).choices[0]
        reply_msg = second.message

    else:
        # Handle non-function queries (e.g., emotional or general statements)
        reply_msg = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rewrite}
            ]
        ).choices[0].message

    answer = (reply_msg.content or "").strip()

    # Keep the system message; trim the rest to MAX_HISTORY
    history.append({"role": "assistant", "content": answer})
    history = [history[0]] + history[-(MAX_HISTORY-1):]

    logging.info(f"Assistant response: {answer}")
    return jsonify({"response": answer})

@app.route("/reset", methods=["POST"])
def reset():
    global history
    history = [{"role": "system", "content": system_prompt}]
    return ("", 204)

if __name__ == '__main__':
    debug_flag = bool(int(os.getenv("DEBUG", "1")))  # default on for local dev
    app.run(host='0.0.0.0', port=5000, debug=debug_flag)
