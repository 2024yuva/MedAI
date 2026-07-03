# MedAI Project Rules

- **Demo Readiness**: The user frequently demos this project to judges. The backend LLM generation MUST always be configured to respond extremely quickly. 
- **Generation Settings**: Ensure that the `PromptBuilder` always instructs the model to give short, 1-2 sentence answers without thinking or reasoning.
- **Ollama Payload**: Keep the `"num_predict"` parameter in the Ollama API payload low (e.g., 100) to guarantee fast inference and prevent long delays during presentations.
