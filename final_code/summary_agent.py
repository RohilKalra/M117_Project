from llama_index.llms.openai import OpenAI


class SummaryAgent:
    def __init__(self, llm: OpenAI, agent_id: int):
        self.llm = llm
        self.agent_id = agent_id

    def summarize(self, text: str, max_tokens: int = 1000) -> str:
        """Generate a summary with a specific focus on maintaining key plot points"""
        prompt = (
            f"Please provide a concise summary of the following text, "
            f"focusing on maintaining the essential plot points and key themes. "
            f"Keep the summary informative but shorter than the original:\n\n{text}"
        )

        response = self.llm.complete(prompt, max_tokens=max_tokens)
        return response.text
