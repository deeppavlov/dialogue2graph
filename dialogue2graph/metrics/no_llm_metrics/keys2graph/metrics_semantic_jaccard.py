from __future__ import annotations
import json
import os
from typing import List, Dict, Tuple, Any

from openai import OpenAI


def evaluate_graph_lists(
    original_graphs: List[Dict[str, Any]],
    generated_graphs: List[Dict[str, Any]],
    output_path: str,
    model_name: str = "o1-mini",
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Сравнивает пары графов (original_graphs[i], generated_graphs[i]) по Semantic-Jaccard метрике,
    но не через эмбеддинги, а через большую языковую модель (LLM).

    Модель получает пару графов, внутри промпта ей даётся задание сопоставить ноды
    и рёбра по смыслу. На выходе модель должна вернуть JSON с полями:
      {
        "matched_nodes": [
          [node1_id, node2_id],
          ...
        ],
        "matched_edges": [
          [src1, tgt1, src2, tgt2],
          ...
        ]
      }

    По этим сопоставлениям мы считаем Semantic-Jaccard:
      jaccard_nodes  = intersection_nodes / union_nodes
      jaccard_edges  = intersection_edges / union_edges

    Результаты сохраняются в `output_path` (JSON), а также возвращаются:
      (результат_по_каждой_паре, средний_j_nodes, средний_j_edges).
    """
    if len(original_graphs) != len(generated_graphs):
        print("Длины списков графов не совпадают — сравнение невозможно.")
        return [], 0.0, 0.0

    results = []
    jaccard_nodes_list = []
    jaccard_edges_list = []

    for idx, (orig_obj, gen_obj) in enumerate(zip(original_graphs, generated_graphs)):
        g1 = orig_obj["graph"]
        g2 = gen_obj["graph"]

        # Сравниваем две структуры через LLM
        comparison_res = _compare_two_graphs_llm(g1, g2, idx, model_name)

        results.append(comparison_res)
        jaccard_nodes_list.append(comparison_res["semantic_jaccard_nodes"])
        jaccard_edges_list.append(comparison_res["semantic_jaccard_edges"])

    # Считаем средние значения
    avg_nodes = (
        sum(jaccard_nodes_list) / len(jaccard_nodes_list) if jaccard_nodes_list else 0.0
    )
    avg_edges = (
        sum(jaccard_edges_list) / len(jaccard_edges_list) if jaccard_edges_list else 0.0
    )

    summary = {
        "avg_semantic_jaccard_nodes": avg_nodes,
        "avg_semantic_jaccard_edges": avg_edges,
        "pairs_count": len(results),
    }

    final_data = {"per_pair": results, "summary": summary}

    # Сохраняем итоговый JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print("\n=== Semantic-Jaccard similarity (avg over all pairs) ===")
    print(f"Nodes: {avg_nodes:.3f}")
    print(f"Edges: {avg_edges:.3f}")
    print(f"Детальные результаты сохранены в {output_path}\n")

    return results, avg_nodes, avg_edges


def _compare_two_graphs_llm(
    g1: Dict[str, Any], g2: Dict[str, Any], pair_index: int, model_name: str = "o1-mini"
) -> Dict[str, Any]:
    """
    Отправляет два диалоговых графа в LLM (o1-mini) и просит сопоставить их ноды и рёбра.
    Возвращает словарь с полями:
      {
        "pair_index": <int>,
        "semantic_jaccard_nodes": <float>,
        "semantic_jaccard_edges": <float>,
        "matched_nodes": [
          {
            "node1_id": ...,
            "node2_id": ...,
            "node1_utterances": [...],
            "node2_utterances": [...]
          },
          ...
        ],
        "matched_edges": [
          {
            "edge1": { "source":..., "target":..., "utterances": [...] },
            "edge2": { "source":..., "target":..., "utterances": [...] }
          },
          ...
        ]
      }
    """
    # Формируем сообщение для LLM.
    # Обратите внимание, что здесь мы предполагаем использование openai.ChatCompletion
    # Если для модели o1-mini нужен другой метод, адаптируйте.

    # "-Match: 'I want a medium size.' and 'I want pepperoni and mushrooms.' if both texts are answers on question 'What size pizza would you like?'\n"
    # "-No Match: 'I want a medium size.' and 'I want pepperoni and mushrooms.' if the first question 'What size pizza would you like?'\n"
    system_message = (
        "You are an assistant that compares two dialogue graphs.\n"
        "You must determine which nodes and edges from Graph1 match those in Graph2 by meaning (goal/dialog act/intent).\n"
        "Not all nodes or edges may have matched pairs. Only include the nodes and edges that have matching pairs.\n"
        "When analyzing, pay attention not only to the text of the nodes or edges in isolation, but also to their context:\n"
        "– for nodes, consider the incoming and outgoing edges;\n"
        "– for edges, consider the nodes they connect.\n"
        "For example, edges with the same text like “yes” may connect completely different nodes, which means they are not matching edges. The surrounding structure—adjacent nodes and edges—can provide more insight into the nature of the element being analyzed.\n"
        "Identify only the similar edges and nodes between the two graphs.\n\n"
        "Examples:\n"
        """1. These nodes should be matched:
- 'I want to go to London.' and 'My destination is New York.' because they have the same meaning, express the same intent, and are responses to similar questions.
- 'Hello! How can I help you book a flight today?' and 'Welcome! How can I assist you with your flight booking today?' → because both utterances initiate the conversation, include a greeting, and ask the same question.
- 'Sure! Which city would you like to travel to?' and 'What is your desired flight destination?' → the same meaning of question
- 'I want to book a flight.' and 'Help me with my flight booking.' → Both utterances are responses to the question about how the assistant can help and express the same intent.
- 'Got it. Any specific toppings you’d prefer?' and 'Please tell me your favorite toppings.' → In both utterances, the assistant is clarifying the user’s topping preferences.
- 'Hi, want to discuss the current weather?' and 'Let's talk about your weather preferences.' → should be mathed if both utterances are starting nodes in the dialog graph and initiate a conversation about the weather.
- 'Stay cozy, and feel free to chat again soon!' and 'It was great to learn about your weather likes!' → should be mathed if both utterances are closing statements in the dialog and serve as a form of farewell.
- 'Hi! Want to chat about new tech gadgets?', 'Let's talk about the latest tech gadgets!', and 'What new gadgets have you heard about recently?' → should be mathed if all of these utterances are starting points in the dialog graph and begin a conversation about gadgets.
- 'I’m curious about the latest smartwatch.' and 'Tell me more about smartwatches.' → because in these utterances the user is asking the assistant to talk about smartwatches in the context of a conversation about gadgets.
- 'Your flight is booked. Is there anything else I can help with?' and 'Here is your booking confirmation.' → In both utterances, the assistant informs the user that the flight has been booked. Although the first one includes an additional question, both serve as closing statements in the dialog graph.

2. These nodes should not be matched:
- 'I like music' and 'My dog is very cute' → These are different parts of the dialogue graph, with different meanings and intents, not related to each other.
- 'Great. Could you let me know your preferred travel dates?' and 'Thank you! I will confirm your booking shortly.' → because the first is a question about travel dates, and the second is a statement about confirming the booking — their meanings are completely different.
- 'All right. I’ve found a suitable flight. Would you like to confirm the booking?' and 'Your flight is booked! Have a great trip!'
→ because the first asks for confirmation, while the second states that the booking has already been made and ends the conversation.
- 'My dates are the 10th to the 15th.' and 'Yes, that works for me.' → because the first responds to a question about travel dates, while the second is a generic positive response to a suggestion — different intents and meanings.
- 'All right. I’ve found a suitable flight. Would you like to confirm the booking?' and 'Your flight has been booked successfully!' → Even though both utterances concern the flight booking process, they reflect different intents and goals. In the first one, the assistant asks the user to confirm the booking, while in the second one, the assistant states that the booking has already been completed.

3.The decision depends on the context:
- 'It can track my sleep and stress levels.' and 'Features like voice control are amazing!' → should be matched if they are answers to the same semantic question and lead the conversation to a similar semantic node. If the utterances (which function as edges) direct the conversation to different semantic areas, then they are not considered similar. For example, if after 'It can track my sleep and stress levels.' the conversation continues about stress tracking, and after 'Features like voice control are amazing!' it continues about voice control features, then they should not be considered close.
- 'Yes, you can add cheese.' and 'No, don’t add cheese.' → These utterances are considered matched if they are responses to a standard question and lead to nodes that are also similar in meaning (e.g., a question about payment methods), and this is information only for slot filling mechanism. However, if these utterances lead to different dialog nodes (e.g., in the first case, the assistant asks about the type of cheese, and in the second case, the assistant asks about payment options), then they should not be considered related.
- 'Hi there! What's your favorite type of weather?' and 'Do you like sunny days or cooler, cloudy weather?' are more likely to be matched than the nodes 'Hi there! What's your favorite type of weather?' and 'Hi, want to discuss the current weather?'. In the second case, even though both utterances contain greetings, they lead to different intents — one asks about the user’s favorite type of weather, while the other asks for an opinion on the current weather. In contrast, in the first case, even though the first utterance includes a greeting and the second does not, they both ask essentially the same question about weather preferences. Furthermore, if the edges following the nodes in the first case are 'I prefer rainy weather.' and 'Sunny days are my favorite.' and they also lead to semantically close nodes, then those edges should be matched as well.
\n
"""
        "Return ONLY valid JSON with the structure:\n"
        "{\n"
        '  "matched_nodes": [ [nodeId_in_graph1, nodeId_in_graph2], ... ],\n'
        '  "matched_edges": [ [src1, tgt1, src2, tgt2], ... ]\n'
        "}\n"
        "Do NOT include any extra keys or text outside this JSON."
    )

    user_prompt = (
        f"Graph1:\n{json.dumps(g1, ensure_ascii=False, indent=2)}\n\n"
        f"Graph2:\n{json.dumps(g2, ensure_ascii=False, indent=2)}\n\n"
        "Please identify semantically equivalent nodes and edges."
    )

    joint_prompt = system_message + "\n" + user_prompt

    messages = [{"role": "user", "content": joint_prompt}]

    print(
        f"\n=== Сравнение пары графов (pair_index={pair_index}) через LLM {model_name} ==="
    )

    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ["OPENAI_API_BASE"]

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        llm_content = completion.choices[0].message.content
    except Exception as e:
        print(f"Ошибка при обращении к модели: {e}")
        # Возвращаем формальный ответ, что сопоставления нет
        return {
            "pair_index": pair_index,
            "semantic_jaccard_nodes": 0.0,
            "semantic_jaccard_edges": 0.0,
            "matched_nodes": [],
            "matched_edges": [],
        }

    # Парсим ответ как JSON

    try:
        parsed = json.loads(llm_content)
    except json.JSONDecodeError:
        try:
            llm_content = llm_content.replace("```", "").replace("json", "")
            parsed = json.loads(llm_content)
        except json.JSONDecodeError:
            print("Не удалось распарсить JSON-ответ от модели.")
            parsed = {"matched_nodes": [], "matched_edges": []}

    matched_nodes_pairs = parsed.get("matched_nodes", [])
    matched_edges_pairs = parsed.get("matched_edges", [])

    # Сразу выведем какие узлы и рёбра модель «соединила»
    matched_nodes_info = []
    for n1_id, n2_id in matched_nodes_pairs:
        node1 = _find_node_by_id(g1, n1_id)
        node2 = _find_node_by_id(g2, n2_id)
        matched_nodes_info.append(
            {
                "node1_id": n1_id,
                "node2_id": n2_id,
                "node1_utterances": node1.get("utterances", []) if node1 else [],
                "node2_utterances": node2.get("utterances", []) if node2 else [],
            }
        )
        print(
            f"Модель сопоставила node {n1_id} [{node1.get('utterances', []) if node1 else '-'}]"
            f" с node {n2_id} [{node2.get('utterances', []) if node2 else '-'}]"
        )

    matched_edges_info = []
    for src1, tgt1, src2, tgt2 in matched_edges_pairs:
        e1 = _find_edge(g1, src1, tgt1)
        e2 = _find_edge(g2, src2, tgt2)
        matched_edges_info.append({"edge1": e1, "edge2": e2})
        print(
            f"Модель сопоставила edge {src1}->{tgt1} [{e1.get('utterances', []) if e1 else '-'}]"
            f" с edge {src2}->{tgt2} [{e2.get('utterances', []) if e2 else '-'}]"
        )

    # Рассчитываем Jaccard для нод
    total_nodes_g1 = len(g1.get("nodes", []))
    total_nodes_g2 = len(g2.get("nodes", []))
    intersection_n = len(matched_nodes_pairs)
    union_n = total_nodes_g1 + total_nodes_g2 - intersection_n
    j_nodes = intersection_n / union_n if union_n else 0.0

    # Рассчитываем Jaccard для рёбер
    total_edges_g1 = len(g1.get("edges", []))
    total_edges_g2 = len(g2.get("edges", []))
    intersection_e = len(matched_edges_pairs)
    union_e = total_edges_g1 + total_edges_g2 - intersection_e
    j_edges = intersection_e / union_e if union_e else 0.0

    result = {
        "pair_index": pair_index,
        "semantic_jaccard_nodes": j_nodes,
        "semantic_jaccard_edges": j_edges,
        "matched_nodes": matched_nodes_info,
        "matched_edges": matched_edges_info,
    }

    print(result)
    return result


def _find_node_by_id(graph: Dict[str, Any], node_id: Any) -> Dict[str, Any]:
    """Возвращает ноду с указанным 'id' из graph['nodes'] или None."""
    for n in graph.get("nodes", []):
        if n.get("id") == node_id:
            return n
    return {}


def _find_edge(graph: Dict[str, Any], src: Any, tgt: Any) -> Dict[str, Any]:
    """Возвращает ребро из graph['edges'] с указанными source и target или None."""
    for e in graph.get("edges", []):
        if e.get("source") == src and e.get("target") == tgt:
            return e
    return {}
