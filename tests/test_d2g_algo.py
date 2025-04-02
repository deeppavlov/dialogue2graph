import pytest
import dotenv
from dialogue2graph.pipelines.core.graph import Graph
from dialogue2graph.pipelines.d2g_algo.pipeline import Pipeline
from dialogue2graph.pipelines.model_storage import ModelStorage

dotenv.load_dotenv()
if not dotenv.find_dotenv():
    pytest.skip("Skipping test as .env file is not found", allow_module_level=True)

ms = ModelStorage()


@pytest.fixture
def sample_dialogues():
    return [
        {
            "messages": [
                {
                    "text": "Hello! Thank you for reaching out about an influencer partnership. How can I assist you today?",
                    "participant": "assistant",
                },
                {"text": "I'd like to learn about your NFT policies.", "participant": "user"},
                {
                    "text": "Certainly! Our approach to NFTs and digital assets is based on transparency and ethical collaborations. We ask influencers to clearly disclose any promotions related to digital assets and follow specific guidelines for creating and distributing content. Do you have any particular questions regarding these policies?",
                    "participant": "assistant",
                },
                {"text": "Actually, can I change my inquiry?", "participant": "user"},
                {"text": "Of course! How would you like to modify your inquiry?", "participant": "assistant"},
                {"text": "I'd like to learn about the partnership benefits instead.", "participant": "user"},
                {
                    "text": "Certainly! Our partnerships offer various benefits including competitive compensation, exclusive access to our digital asset platforms, and opportunities for co-branded campaigns. We support our influencers with dedicated resources to maximize the impact of their content. Would you like more details or to proceed with an application?",
                    "participant": "assistant",
                },
                {"text": "Actually, I have another question about policies.", "participant": "user"},
                {
                    "text": "Certainly! Our approach to NFTs and digital assets is based on transparency and ethical collaborations. We ask influencers to clearly disclose any promotions related to digital assets and follow specific guidelines for creating and distributing content. Do you have any particular questions regarding these policies?",
                    "participant": "assistant",
                },
                {"text": "Can you provide more details on content guidelines?", "participant": "user"},
                {
                    "text": "Certainly! For content creation, we require that any digital asset promotions clearly state the nature of the asset, its authenticity, and any associated benefits or risks. Additionally, all distribution channels must comply with relevant regulations and our own standards for quality and accuracy. Would you like to discuss the partnership process next?",
                    "participant": "assistant",
                },
                {"text": "Yes, let's discuss the partnership process.", "participant": "user"},
                {
                    "text": "Our partnership process involves several steps: initial proposal review, agreement on terms, content planning, and execution. We collaborate closely with our influencers to ensure that the content aligns with both parties' goals and complies with our NFT policies. Would you like to proceed with a partnership application?",
                    "participant": "assistant",
                },
                {"text": "I'd like to proceed with the application.", "participant": "user"},
                {
                    "text": "Great! Please fill out our partnership application form on our website, and our team will review it within 5 business days. Once approved, we'll reach out to discuss the next steps. Is there anything else I can help you with today?",
                    "participant": "assistant",
                },
                {"text": "No, that's all. Thank you.", "participant": "user"},
                {"text": "You're welcome! We're excited about the possibility of partnering with you. Have a great day!", "participant": "assistant"},
            ],
        },
        {
            "messages": [
                {
                    "text": "Hello! Thank you for reaching out about an influencer partnership. How can I assist you today?",
                    "participant": "assistant",
                },
                {"text": "I'd like to learn about your NFT policies.", "participant": "user"},
                {
                    "text": "Certainly! Our approach to NFTs and digital assets is based on transparency and ethical collaborations. We ask influencers to clearly disclose any promotions related to digital assets and follow specific guidelines for creating and distributing content. Do you have any particular questions regarding these policies?",
                    "participant": "assistant",
                },
                {"text": "Actually, can I change my inquiry?", "participant": "user"},
                {"text": "Of course! How would you like to modify your inquiry?", "participant": "assistant"},
                {"text": "I'd like to learn about the partnership benefits instead.", "participant": "user"},
                {
                    "text": "Certainly! Our partnerships offer various benefits including competitive compensation, exclusive access to our digital asset platforms, and opportunities for co-branded campaigns. We support our influencers with dedicated resources to maximize the impact of their content. Would you like more details or to proceed with an application?",
                    "participant": "assistant",
                },
                {"text": "Actually, I have another question about policies.", "participant": "user"},
                {
                    "text": "Of course! Our NFT and digital asset policies ensure that all collaborations are transparent and ethical. We require influencers to clearly disclose any digital asset promotions and adhere to specific guidelines regarding content creation and distribution. Do you have any specific questions about these policies?",
                    "participant": "assistant",
                },
                {"text": "Can you provide more details on content guidelines?", "participant": "user"},
                {
                    "text": "Certainly! For content creation, we require that any digital asset promotions clearly state the nature of the asset, its authenticity, and any associated benefits or risks. Additionally, all distribution channels must comply with relevant regulations and our own standards for quality and accuracy. Would you like to discuss the partnership process next?",
                    "participant": "assistant",
                },
                {"text": "Yes, let's discuss the partnership process.", "participant": "user"},
                {
                    "text": "Our partnership process involves several steps: initial proposal review, agreement on terms, content planning, and execution. We collaborate closely with our influencers to ensure that the content aligns with both parties' goals and complies with our NFT policies. Would you like to proceed with a partnership application?",
                    "participant": "assistant",
                },
                {"text": "I'd like to proceed with the application.", "participant": "user"},
                {
                    "text": "Great! Please fill out our partnership application form on our website, and our team will review it within 5 business days. Once approved, we'll reach out to discuss the next steps. Is there anything else I can help you with today?",
                    "participant": "assistant",
                },
                {"text": "No, that's all. Thank you.", "participant": "user"},
                {"text": "You're welcome! We're excited about the possibility of partnering with you. Have a great day!", "participant": "assistant"},
            ],
        },
        {
            "messages": [
                {
                    "text": "Hello! Thank you for reaching out about an influencer partnership. How can I assist you today?",
                    "participant": "assistant",
                },
                {"text": "I'd like to learn about your NFT policies.", "participant": "user"},
                {
                    "text": "Of course! Our NFT and digital asset policies ensure that all collaborations are transparent and ethical. We require influencers to clearly disclose any digital asset promotions and adhere to specific guidelines regarding content creation and distribution. Do you have any specific questions about these policies?",
                    "participant": "assistant",
                },
                {"text": "Actually, can I change my inquiry?", "participant": "user"},
                {"text": "Of course! How would you like to modify your inquiry?", "participant": "assistant"},
                {"text": "I'd like to learn about the partnership benefits instead.", "participant": "user"},
                {
                    "text": "Certainly! Our partnerships offer various benefits including competitive compensation, exclusive access to our digital asset platforms, and opportunities for co-branded campaigns. We support our influencers with dedicated resources to maximize the impact of their content. Would you like more details or to proceed with an application?",
                    "participant": "assistant",
                },
                {"text": "Actually, I have another question about policies.", "participant": "user"},
                {
                    "text": "Certainly! Our approach to NFTs and digital assets is based on transparency and ethical collaborations. We ask influencers to clearly disclose any promotions related to digital assets and follow specific guidelines for creating and distributing content. Do you have any particular questions regarding these policies?",
                    "participant": "assistant",
                },
                {"text": "Can you provide more details on content guidelines?", "participant": "user"},
                {
                    "text": "Certainly! For content creation, we require that any digital asset promotions clearly state the nature of the asset, its authenticity, and any associated benefits or risks. Additionally, all distribution channels must comply with relevant regulations and our own standards for quality and accuracy. Would you like to discuss the partnership process next?",
                    "participant": "assistant",
                },
                {"text": "Yes, let's discuss the partnership process.", "participant": "user"},
                {
                    "text": "Our partnership process involves several steps: initial proposal review, agreement on terms, content planning, and execution. We collaborate closely with our influencers to ensure that the content aligns with both parties' goals and complies with our NFT policies. Would you like to proceed with a partnership application?",
                    "participant": "assistant",
                },
                {"text": "I'd like to proceed with the application.", "participant": "user"},
                {
                    "text": "Great! Please fill out our partnership application form on our website, and our team will review it within 5 business days. Once approved, we'll reach out to discuss the next steps. Is there anything else I can help you with today?",
                    "participant": "assistant",
                },
                {"text": "No, that's all. Thank you.", "participant": "user"},
                {"text": "You're welcome! We're excited about the possibility of partnering with you. Have a great day!", "participant": "assistant"},
            ],
        },
        {
            "messages": [
                {
                    "text": "Hello! Thank you for reaching out about an influencer partnership. How can I assist you today?",
                    "participant": "assistant",
                },
                {"text": "I'd like to learn about your NFT policies.", "participant": "user"},
                {
                    "text": "Certainly! Our approach to NFTs and digital assets is based on transparency and ethical collaborations. We ask influencers to clearly disclose any promotions related to digital assets and follow specific guidelines for creating and distributing content. Do you have any particular questions regarding these policies?",
                    "participant": "assistant",
                },
                {"text": "Can you provide more details on content guidelines?", "participant": "user"},
                {
                    "text": "Certainly! For content creation, we require that any digital asset promotions clearly state the nature of the asset, its authenticity, and any associated benefits or risks. Additionally, all distribution channels must comply with relevant regulations and our own standards for quality and accuracy. Would you like to discuss the partnership process next?",
                    "participant": "assistant",
                },
                {"text": "Actually, I want to change my question.", "participant": "user"},
                {"text": "Of course! How would you like to modify your inquiry?", "participant": "assistant"},
                {"text": "I'd like to learn about the partnership benefits instead.", "participant": "user"},
                {
                    "text": "Certainly! Our partnerships offer various benefits including competitive compensation, exclusive access to our digital asset platforms, and opportunities for co-branded campaigns. We support our influencers with dedicated resources to maximize the impact of their content. Would you like more details or to proceed with an application?",
                    "participant": "assistant",
                },
                {"text": "I'd like to proceed with the application.", "participant": "user"},
                {
                    "text": "Great! Please fill out our partnership application form on our website, and our team will review it within 5 business days. Once approved, we'll reach out to discuss the next steps. Is there anything else I can help you with today?",
                    "participant": "assistant",
                },
                {"text": "No, that's all. Thank you.", "participant": "user"},
                {"text": "You're welcome! We're excited about the possibility of partnering with you. Have a great day!", "participant": "assistant"},
            ],
        },
    ]


def test_d2g_algo(sample_dialogues):
    """Test that graph is generated without errors"""

    ms.add(
        key="filling_llm",
        config={"name": "o3-mini", "temperature": 1},
        model_type="llm",
    )
    ms.add(
        key="formatting_llm",
        config={"name": "gpt-4o-mini", "temperature": 0},
        model_type="llm",
    )
    ms.add(
        key="sim_model",
        config={"model_name": "BAAI/bge-m3", "device": "cpu"},
        model_type="emb",
    )

    pipeline = Pipeline(model_storage=ms, filling_llm="filling_llm", formatting_llm="formatting_llm", sim_model="sim_model")

    graph = pipeline.invoke(sample_dialogues)

    assert isinstance(graph, Graph)
