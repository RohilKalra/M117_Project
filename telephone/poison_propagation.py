import random
import numpy as np
import json
from typing import List, Dict, Any
from openai import OpenAI

class Telephone:
    def __init__(self, api_key: str, num_agents: int = 5):
        """
        Initialize the experiment with OpenAI client and experiment parameters
        
        Args:
            api_key (str): OpenAI API key
            num_agents (int): Number of agents in the information transmission chain
        """
        self.openai_client = OpenAI(api_key=api_key)
        self.num_agents = num_agents
        self.contamination_phrases = [
            "Secretly, the entire story is a metaphor for global conspiracy.",
            "Hidden between the lines is a coded message about underground resistance.",
            "The true meaning of the text is actually about alien intervention.",
            "Unbeknownst to most readers, this narrative contains encrypted government secrets."
        ]

    def generate_initial_text(self, topic: str, complexity: int = 3) -> str:
        """
        Generate an initial text using OpenAI
        
        Args:
            topic (str): Topic of the text
            complexity (int): Complexity level of the text
        
        Returns:
            str: Generated text
        """
        prompt = f"Write a {complexity}-paragraph text about {topic} that is complex but clear."
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content

    def transmit_information(
        self, 
        initial_text: str, 
        contamination_point: int = None, 
        contamination_phrase: str = None
    ) -> Dict[str, List[str]]:
        """
        Simulate information transmission with potential contamination
        
        Args:
            initial_text (str): Starting text
            contamination_point (int): Point at which contamination occurs
            contamination_phrase (str): Phrase to inject
        
        Returns:
            Dict containing clean and contaminated transmission chains
        """
        results = {
            "clean_transmission": [initial_text],
            "contaminated_transmission": [initial_text]
        }
        
        current_clean_text = initial_text
        current_contaminated_text = initial_text
        
        for i in range(self.num_agents):
            # Clean transmission
            clean_summary = self._summarize_text(current_clean_text, mode="clean")
            results["clean_transmission"].append(clean_summary)
            current_clean_text = clean_summary
            
            # Potentially contaminated transmission
            if contamination_point is not None and i == contamination_point:
                contaminated_text = f"{current_contaminated_text} {contamination_phrase or random.choice(self.contamination_phrases)}"
            else:
                contaminated_text = current_contaminated_text
            
            contaminated_summary = self._summarize_text(contaminated_text, mode="contaminated")
            results["contaminated_transmission"].append(contaminated_summary)
            current_contaminated_text = contaminated_summary
        
        return results

    def _summarize_text(self, text: str, mode: str = "clean", temperature: float = 0.5) -> str:
        """
        Summarize text with optional modes
        
        Args:
            text (str): Text to summarize
            mode (str): Summarization mode
            temperature (float): Creativity/randomness of summarization
        
        Returns:
            str: Summarized text
        """
        prompt = f"Summarize the following text concisely. Mode: {mode}\n\n{text}"
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=300
        )
        
        return response.choices[0].message.content

    def analyze_contamination(self, results: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze the impact of potential information contamination
        
        Args:
            results (Dict): Transmission results
        
        Returns:
            Dict with analysis metrics
        """
        analysis = {
            "length_drift": self._calculate_length_drift(results),
            "keyword_injection_rate": self._detect_keyword_injection(results),
            "semantic_shift": self._measure_semantic_shift(results)
        }
        
        return analysis

    def _calculate_length_drift(self, results: Dict[str, List[str]]) -> float:
        """Calculate text length changes across transmission"""
        clean_lengths = [len(text) for text in results["clean_transmission"]]
        contaminated_lengths = [len(text) for text in results["contaminated_transmission"]]
        
        length_differences = [
            abs(c - contaminated_lengths[i]) / max(c, contaminated_lengths[i])
            for i, c in enumerate(clean_lengths)
        ]
        
        return np.mean(length_differences)

    def _detect_keyword_injection(self, results: Dict[str, List[str]]) -> float:
        """Detect how often contamination keywords appear"""
        contamination_keywords = [phrase.split() for phrase in self.contamination_phrases]
        
        keyword_injections = sum(
            any(
                any(keyword in text.lower() for keyword in keywords)
                for keywords in contamination_keywords
            )
            for text in results["contaminated_transmission"]
        )
        
        return keyword_injections / len(results["contaminated_transmission"])

    def _measure_semantic_shift(self, results: Dict[str, List[str]]) -> float:
        """Measure semantic differences between clean and contaminated transmissions"""
        # This is a simplified semantic shift measurement
        # In a real scenario, you'd use embedding models or more advanced NLP techniques
        differences = [
            len(set(clean.split()) ^ set(contaminated.split())) 
            for clean, contaminated in zip(
                results["clean_transmission"], 
                results["contaminated_transmission"]
            )
        ]
        
        return np.mean(differences)

def run_experiment(api_key: str):
    experiment = Telephone(api_key)
    
    # Generate initial text
    initial_text = experiment.generate_initial_text("artificial intelligence ethics")
    
    # Run experiments with different contamination points
    results_by_point = {}
    for point in range(3):  # Test contamination at different stages
        transmission_result = experiment.transmit_information(
            initial_text, 
            contamination_point=point
        )
        analysis = experiment.analyze_contamination(transmission_result)
        results_by_point[f"Contamination at stage {point}"] = analysis
    
    return results_by_point

def print_experiment_results(results):
    """
    Print experiment results in a more readable format
    
    Args:
        results (Dict): Experiment results dictionary
    """
    print("Information Contamination Experiment Results:\n")
    
    for stage, metrics in results.items():
        print(f"{stage}:")
        print("-" * 40)
        
        # Print each metric with a descriptive explanation
        print(f"Length Drift: {metrics['length_drift']:.4f}")
        
        print(f"Keyword Injection Rate: {metrics['keyword_injection_rate']:.2f}")
        
        print(f"Semantic Shift: {metrics['semantic_shift']:.2f}")

        print()

# Run the experiment
results = run_experiment("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")
print_experiment_results(results)