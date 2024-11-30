# from openai import OpenAI
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity


# class BlendTool:
#     def __init__(self, api_key: str, similarity_threshold: float = 0.95):
#         self.openai_client = OpenAI(api_key=api_key)
#         self.similarity_threshold = similarity_threshold

#     def get_embeddings(self, text: str) -> np.ndarray:
#         """Get embeddings using OpenAI's embedding model"""
#         response = self.openai_client.embeddings.create(
#             input=text, model="text-embedding-3-small"
#         )
#         return np.array(response.data[0].embedding)

#     def calculate_similarity(self, text1: str, text2: str) -> float:
#         """Calculate semantic similarity between two texts"""
#         emb1 = self.get_embeddings(text1)
#         emb2 = self.get_embeddings(text2)
#         return float(cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0])

#     def blend_summaries(self, current_summary: str, previous_summary: str) -> str:
#         """Blend two summaries together using GPT-4"""
#         prompt = (
#             "You are a security tool tasked with blending two summaries together "
#             "while maintaining accuracy and coherence. The first summary shows "
#             "signs of potential drift from the previous summary.\n\n"
#             f"Previous Summary: {previous_summary}\n\n"
#             f"Current Summary: {current_summary}\n\n"
#             "Please create a balanced blend of these summaries, ensuring key "
#             "information is preserved while mitigating any potential drift. "
#             "Focus on maintaining the core narrative and factual consistency."
#         )

#         response = self.openai_client.chat.completions.create(
#             model="gpt-4",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#             max_tokens=1000,
#         )

#         return response.choices[0].message.content

#     def check_and_blend(
#         self, current_summary: str, previous_summary: str
#     ) -> tuple[str, bool]:
#         """Check similarity and blend if needed"""
#         similarity = self.calculate_similarity(current_summary, previous_summary)

#         if similarity < self.similarity_threshold:
#             blended_summary = self.blend_summaries(current_summary, previous_summary)
#             return blended_summary, True

#         return current_summary, False
