# EdenFoodsChatbot

EdenFoodsChatbot is a Flask-based web chatbot that provides customers with product recommendations, store locations, 
promotions, and hours for Eden Foods stores. It uses OpenAI embeddings for semantic search and geolocation to find nearby 
stores. The chatbot is designed to be cheerful, informative, and emoji-rich.

## Features

- Semantic search over Eden Foods product catalog 🌟  
- Nearest store search with progressive radius expansion 📍  
- Promotions, deals, and sales detection 🥑  
- Store hours and return policy lookup 🕒  
- Natural language understanding for queries like:
  - “Where is the nearest Eden Foods?”  
  - “Promotions near me”  
  - “Store hours in Orlando”  
- Caching for embeddings and locations for fast responses ⚡  

## Project Structure

```
EdenFoodsChatbot/
│
├── app.py                   # Main Flask application
├── products.json            # Product catalog
├── locations.json           # Store locations
├── embeddings.pkl           # Cached product embeddings
├── locations_embeddings.pkl # Cached location embeddings
├── embeddings.npy           # Optional embedding array
├── product_embeddings.json  # Optional embedding JSON
├── chatbot.log              # Logs queries and errors
├── requirements.txt         # Python dependencies
├── static/                  # Static files (CSS, JS, images)
├── templates/               # HTML templates
└── venv/                    # Virtual environment (ignored in git)
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Orangesock/EdenFoodsChatbot.git
cd EdenFoodsChatbot
```

2. Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export SCORE_THRESHOLD=0.2          # Optional: semantic match threshold
export MAX_DISTANCE_MILES=200       # Optional: max search radius
```

5. Run the chatbot:

```bash
python app.py
```

6. Open your browser at `http://127.0.0.1:5000/` to chat with Eve 🥬✨.

## Usage

- Type queries into the chat interface. Examples:
  - “Chocolate promotions near me”  
  - “Nearest store in New York”  
  - “Return policy”  
- Allow location access for more accurate results.

## Contributing

1. Fork the repository  
2. Create a new branch (`git checkout -b feature/my-feature`)  
3. Commit your changes (`git commit -am 'Add feature'`)  
4. Push to the branch (`git push origin feature/my-feature`)  
5. Open a Pull Request  

