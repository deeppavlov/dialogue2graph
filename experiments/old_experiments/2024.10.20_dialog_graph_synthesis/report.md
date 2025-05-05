# Cycle Graph Generation and Augmentation of Synthetic Dialogs

## Issues and Goals

The main challenge was creating logically coherent and diverse synthetic dialogs at scale for tasks like chatbot training. The approach uses a cycle graph structure where dialogs must logically return to their starting point, enabling continuous conversation flows.

## Key Challenges and Limitations

1. **Cycle Implementation Difficulty**
   - Creating natural dialog cycles proved challenging
   - The return point to complete the cycle often feels forced or artificial
   - Current implementation tends to follow a linear path until the final cycling point
   - The dialog structure is tightly coupled to the example provided in the prompt

2. **Prompt Dependency**
   - Generated dialogs closely mirror the structure of the example in the prompt
   - Diversification is currently limited by this structural coupling
   - Different dialog patterns require different example structures in the prompt

## Implementation Approach

### 1. Cycle Graph Generation
- GPT-4 mini was unable to generate valid cycle graphs consistently
- Requires larger models like GPT-4 to create JSON-structured dialog graphs
- Enforces strict rules:
  - Must be cyclic with last edge connecting back to first node
  - Each node contains assistant utterances
  - Each edge contains user utterances  
  - Transitions must be logically coherent
  - Utterances in same list must be synonymous but differently worded

### 2. Graph Validation 
- Validates each transition triplet using GPT-4 mini
- Checks if source -> edge -> target transitions make logical sense
- Returns validation results with explanations of any issues
- Maps node IDs and validates their existence

### 3. Dialog Sampling
- Uses DialogSampler to generate complete dialogs from valid graphs
- Maintains conversation flow by following graph paths
- Currently limited in effectiveness due to linear graph structure
- Sampling variations are minimal since the graph follows a mostly linear path

### 4. Dialog Augmentation
- Takes sampled dialogs and generates variations
- For each utterance in the dialog:
  - Generates a single alternative utterance
  - Maintains the same meaning/intent
  - Uses different phrasing while keeping theme consistency
  - Ensures natural conversation flow
- Provides one-to-one utterance replacement options

## Differentiation Capabilities

- While structural variation is limited, the system can handle different topics
- Topic variations provide some dialog diversity
- Augmentation helps create alternative phrasings for each utterance
- The core conversation flow remains similar across topics

## Results

The implementation demonstrates:
- A functional cycle graph generation pipeline with validation
- A dialog sampling system (though limited by linear structure)
- A one-to-one utterance augmentation framework
- Successfully handles different conversation topics while maintaining structure

## Future Improvements

1. Integration into the main library
2. Expanding prompt examples to enable more diverse dialog structures
3. Improving cycle implementation for more natural conversation loops
4. Reducing prompt dependency to allow more structural variation
5. Enhancing sampling capabilities to better utilize graph structure
6. Expanding augmentation to potentially generate multiple alternatives per utterance

## Technical Recommendations

1. Experiment with multiple example structures in prompts
2. Implement branching paths before cycle points
3. Develop more natural transition points for cycling
4. Create topic-specific templates for better dialog coherence