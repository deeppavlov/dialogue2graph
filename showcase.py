from dotenv import load_dotenv
from dialogue2graph.pipelines.topic_generation import TopicGenerationPipeline

load_dotenv()

pipeline = TopicGenerationPipeline()

pipeline.invoke("buying a house with a cash or a credit card, but buyer cannot have debts",
                "house_purchase.json")
