import os
from dotenv import load_dotenv

load_dotenv()

# Force the offline stubs so pytest never reaches Ollama / Neo4j. These are
# assigned, not setdefault-ed: load_dotenv() above picks up cbt_kg/.env, whose
# real-deployment values (EXTRACTOR=local, GENERATOR=local) would otherwise win
# and quietly point the whole suite at a live Ollama.
os.environ["EXTRACTOR"] = "stub"
os.environ["GENERATOR"] = "echo"
os.environ["GRAPH_BACKEND"] = "memory"
