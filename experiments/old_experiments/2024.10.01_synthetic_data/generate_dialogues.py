from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
from dotenv import load_dotenv
import os
from tqdm import tqdm
import time

load_dotenv() 

with open('experiments/2024.10.01_synthetic_data/prompt.txt', 'r') as f:
    prompt = HumanMessage(f.read())

history = [prompt]
model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0)
def generate_dialogue_graph_pair():
    response = model.invoke(history)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        print("Error: Invalid JSON response. Retrying...")
        return None
    

def main():
    results = []
    for i in tqdm(range(100)):
        pair = None
        while pair is None:
            pair = generate_dialogue_graph_pair()
            if pair is not None:
                pair['id'] = i + 1
                results.append(pair)
            else:
                print("Invalid JSON.")
        time.sleep(1)  # Add a small delay between requests to avoid rate limiting

    with open('experiments/2024.10.01_synthetic_data/generated_data/dialogue_graph_pairs.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Generation complete. Results saved to dialogue_graph_pairs.json")

if __name__ == "__main__":
    main()