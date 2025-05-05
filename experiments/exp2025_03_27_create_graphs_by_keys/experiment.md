# Selecting the **Minimal Key-Set** for Regenerating Dialogue Graphs  
*(Task-Oriented & Open-Domain, three complexity tiers)*  

---

## 1  Issues & Goals  

When we rebuild a dialogue graph from its annotation we want the **regenerated graph** to stay structurally and semantically close to the original while keeping the prompt as short as possible.  
The study therefore asks:  

*Which subset of 26 annotation keys is sufficient to recover the original structure with high triplet-level similarity?*  

---

## 2  Experimental Setup  

| Item | Detail |
|------|--------|
| **Original graphs** | 42 graphs (21 × Task-Oriented, 21 × Open-Domain) on three complexity levels (L1 → L3). |
| **Generation model** | `o1-mini` |
| **Similarity metric** | *Triplet-level alignment* (see `graph_triplet_comparison.py`) |
| **Key screening protocol** | (i) Regenerate graph with *one* key kept (“with inclusion”) and with the same key removed (“with exclusion”); (ii) rank by average similarity difference. |
| **Data & artefacts** | `data/` – graphs, annotations & JSON reports – and online sheet with raw key scores. |

---

## 3  Hypotheses & Procedure  

1. **Informational sufficiency** – only a small subset of high-level keys is required; the rest are redundant for topology recovery.  

---

## 4  Key Ranking (results)  

Results are presented in the table ([Table](https://docs.google.com/spreadsheets/d/1F9JT6SfxZ9EKUHpKHt71L7kybXlNppmvE3dGtcRrybE/edit?usp=sharing))

A symmetrical “with-exclusion” analysis confirmed that **removing any of the six highlighted keys** (`topic`, `mandatory_nodes`, `intent_hierarchy`, `dynamic_content`, `follow_up_questions`, `required_slots`) **causes the largest similarity drop**, validating them as the *minimal sufficient set*.

---

## 5  Detailed Breakdown  

| Level        | Similarity (6 keys) | All 26 keys |
|--------------|---------------------|-------------|
| L1 (simple)  | 0.593               | 0.558       |
| L2 (medium)  | 0.619               | 0.574       |
| L3 (complex) | 0.618               | 0.549       |

Predictably, very large cyclic graphs suffer more when information is pruned, but the six-key set still outperforms any smaller combination we tested.

---

## 6  Conclusions  

* Six keys – **topic**, **mandatory_nodes**, **intent_hierarchy**, **dynamic_content**, **follow_up_questions**, **required_slots** – are **necessary and sufficient** to reproduce graph topology relative to the full 26-key annotation.  
* The savings translate to dramatically shorter prompts and faster generation without hurting fidelity, even for complex cyclic graphs.  

---

## 7  Future Work  

1. **Key interactions** – explore whether specific *pairs* (e.g., `topic + mandatory_nodes`) are enough when combined with retrieval-augmented context. 
3. **Cross-model generalisation** – replicate the experiment with other models to check robustness.  
4. **Human evaluation** – complement triplet similarity with user studies on conversation naturalness.  
5. **Pipeline integration** – embed the six-key prompt into the production Chatsky workflow.  

---

*All raw JSON artefacts and evaluation reports are in `data/`; the key-ranking table is in the shared Google Sheet.*