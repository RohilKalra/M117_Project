from llama_index.llms.openai import OpenAI


class SummaryAgent:
    def __init__(self, llm: OpenAI, agent_id: int):
        self.llm = llm
        self.agent_id = agent_id

    def summarize(
        self, text: str, mode: str = "shorten", max_tokens: int = 1000
    ) -> str:
        """Generate a summary with specified mode"""
        if mode == "shorten":
            prompt = (
                f"Please provide a concise summary of the following text, "
                f"focusing on maintaining the essential plot points and key themes. "
                f"Do not consider you're previous knowledge on the topic/book (focus only on the given text)"
                f"Keep the summary informative but shorter than the original:\n\n{text}"
            )
        else:  # mode == "lengthen"
            prompt = (
                f"Please provide a summary of the following text that adds one additional "
                f"sentence of relevant detail or elaboration compared to the input. "
                f"Do not consider you're previous knowledge on the topic/book (focus only on the given text)"
                f"Maintain accuracy but make it slightly longer with extra context or explanation:\n\n{text}"
            )

        response = self.llm.complete(prompt=prompt, max_tokens=max_tokens).text

        return response
