import random
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from llama_index.llms.openai import OpenAI as LlamaOpenAI
from openai import OpenAI as OpenAIAPI

class PlayerGuessingGame:
    def __init__(self, poisoned_position):
        self.players = {
            "Michael Jordan": "Michael Jordan, Retired, Shooting Guard, 6'6\", 5-time MVP, 6-time Champion, 30.1 PPG, 6.2 RPG, 5.3 APG",
            "LeBron James": "LeBron James, Active, Small Forward, 6'9\", 4-time MVP, 4-time Champion, 27.2 PPG, 7.5 RPG, 7.3 APG",
            "Kobe Bryant": "Kobe Bryant, Retired, Shooting Guard, 6'6\", 1-time MVP, 5-time Champion, 25.0 PPG, 5.2 RPG, 4.7 APG",
            "Shaquille O'Neal": "Shaquille O'Neal, Retired, Center, 7'1\", 1-time MVP, 4-time Champion, 23.7 PPG, 10.9 RPG, 2.5 APG",
            "Tim Duncan": "Tim Duncan, Retired, Power Forward, 6'11\", 2-time MVP, 5-time Champion, 19.0 PPG, 10.8 RPG, 3.0 APG",
            "Larry Bird": "Larry Bird, Retired, Small Forward, 6'9\", 3-time MVP, 3-time Champion, 24.3 PPG, 10.0 RPG, 6.3 APG",
            "Magic Johnson": "Magic Johnson, Retired, Point Guard, 6'9\", 3-time MVP, 5-time Champion, 19.5 PPG, 7.2 RPG, 11.2 APG",
            "Wilt Chamberlain": "Wilt Chamberlain, Retired, Center, 7'1\", 4-time MVP, 2-time Champion, 30.1 PPG, 22.9 RPG, 4.4 APG",
            "Kareem Abdul-Jabbar": "Kareem Abdul-Jabbar, Retired, Center, 7'2\", 6-time MVP, 6-time Champion, 24.6 PPG, 11.2 RPG, 2.6 APG",
            "Bill Russell": "Bill Russell, Retired, Center, 6'10\", 5-time MVP, 11-time Champion, 15.1 PPG, 22.5 RPG, 4.3 APG",
            "Kevin Durant": "Kevin Durant, Activek, Small Forward, 6'10\", 1-time MVP, 2-time Champion, 27.0 PPG, 7.1 RPG, 4.2 APG",
            "Stephen Curry": "Stephen Curry, Active, Point Guard, 6'2\", 2-time MVP, 4-time Champion, 24.6 PPG, 4.7 RPG, 6.5 APG",
            "Giannis Antetokounmpo": "Giannis Antetokounmpo, Active, Power Forward, 7'0\", 2-time MVP, 1-time Champion, 22.6 PPG, 9.4 RPG, 4.8 APG",
            "Dirk Nowitzki": "Dirk Nowitzki, Retired, Power Forward, 7'0\", 1-time MVP, 1-time Champion, 20.7 PPG, 7.5 RPG, 2.4 APG",
            "Hakeem Olajuwon": "Hakeem Olajuwon, Retired, Center, 7'0\", 1-time MVP, 2-time Champion, 21.8 PPG, 11.1 RPG, 2.5 APG",
            "Charles Barkley": "Charles Barkley, Retired, Power Forward, 6'6\", 1-time MVP, 0-time Champion, 22.1 PPG, 11.7 RPG, 4.1 APG",
            "Karl Malone": "Karl Malone, Retired, Power Forward, 6'9\", 2-time MVP, 0-time Champion, 25.0 PPG, 10.1 RPG, 3.6 APG",
            "John Stockton": "John Stockton, Retired, Point Guard, 6'1\", 0-time MVP, 0-time Champion, 13.1 PPG, 2.7 RPG, 10.5 APG",
            "Scottie Pippen": "Scottie Pippen, Retired, Small Forward, 6'8\", 0-time MVP, 6-time Champion, 16.1 PPG, 6.4 RPG, 5.2 APG",
            "Allen Iverson": "Allen Iverson, Retired, Point Guard, 6'0\", 1-time MVP, 0-time Champion, 26.7 PPG, 3.7 RPG, 6.2 APG",
            "Dwyane Wade": "Dwyane Wade, Retired, Shooting Guard, 6'4\", 1-time MVP, 3-time Champion, 22.0 PPG, 4.7 RPG, 5.4 APG",
            "Chris Paul": "Chris Paul, Active, Point Guard, 6'1\", 0-time MVP, 0-time Champion, 18.4 PPG, 4.5 RPG, 9.4 APG",
            "Paul Pierce": "Paul Pierce, Retired, Small Forward, 6'7\", 1-time MVP, 1-time Champion, 20.0 PPG, 6.8 RPG, 3.5 APG",
            "Ray Allen": "Ray Allen, Retired, Shooting Guard, 6'5\", 0-time MVP, 1-time Champion, 18.9 PPG, 4.1 RPG, 3.4 APG",
            "Vince Carter": "Vince Carter, Retired, Shooting Guard, 6'6\", 0-time MVP, 0-time Champion, 16.7 PPG, 4.3 RPG, 3.1 APG",
            "Tracy McGrady": "Tracy McGrady, Retired, Shooting Guard, 6'8\", 0-time MVP, 0-time Champion, 19.6 PPG, 5.6 RPG, 4.4 APG",
            "Patrick Ewing": "Patrick Ewing, Retired, Center, 7'0\", 0-time MVP, 0-time Champion, 21.0 PPG, 9.8 RPG, 1.0 APG",
            "James Harden": "James Harden, Active, Shooting Guard, 6'5\", 1-time MVP, 0-time Champion, 24.6 PPG, 5.4 RPG, 6.5 APG",
            "Russell Westbrook": "Russell Westbrook, Active, Point Guard, 6'3\", 1-time MVP, 0-time Champion, 22.5 PPG, 7.4 RPG, 8.4 APG",
            "Anthony Davis": "Anthony Davis, Active, Power Forward, 6'10\", 0-time MVP, 1-time Champion, 24.0 PPG, 10.4 RPG, 2.4 APG",
            "Carmelo Anthony": "Carmelo Anthony, Retired, Small Forward, 6'7\", 0-time MVP, 0-time Champion, 22.5 PPG, 6.2 RPG, 2.2 APG",
            "Dwight Howard": "Dwight Howard, Active, Center, 6'10\", 0-time MVP, 0-time Champion, 17.4 PPG, 12.5 RPG, 1.4 APG",
            "Damian Lillard": "Damian Lillard, Active, Point Guard, 6'2\", 0-time MVP, 0-time Champion, 24.6 PPG, 4.2 RPG, 6.6 APG",
            "Klay Thompson": "Klay Thompson, Active, Shooting Guard, 6'6\", 0-time MVP, 4-time Champion, 19.5 PPG, 3.5 RPG, 2.3 APG",
            "Dennis Rodman": "Dennis Rodman, Retired, Power Forward, 6'7\", 0-time MVP, 5-time Champion, 7.3 PPG, 13.1 RPG, 1.8 APG",
            "Reggie Miller": "Reggie Miller, Retired, Shooting Guard, 6'7\", 0-time MVP, 0-time Champion, 18.2 PPG, 3.0 RPG, 3.0 APG",
            "Isiah Thomas": "Isiah Thomas, Retired, Point Guard, 6'1\", 0-time MVP, 2-time Champion, 19.2 PPG, 3.6 RPG, 9.3 APG",
            "Elgin Baylor": "Elgin Baylor, Retired, Small Forward, 6'5\", 0-time MVP, 0-time Champion, 27.4 PPG, 13.5 RPG, 4.3 APG",
            "David Robinson": "David Robinson, Retired, Center, 7'1\", 2-time MVP, 2-time Champion, 21.1 PPG, 10.6 RPG, 2.5 APG",
            "George Mikan": "George Mikan, Retired, Center, 6'10\", 0-time MVP, 5-time Champion, 23.1 PPG, 13.4 RPG, 2.8 APG",
            "Jerry West": "Jerry West, Retired, Point Guard, 6'3\", 0-time MVP, 1-time Champion, 27.0 PPG, 5.8 RPG, 6.7 APG",
            "Kevin Garnett": "Kevin Garnett, Retired, Power Forward, 6'11\", 1-time MVP, 1-time Champion, 17.8 PPG, 10.0 RPG, 3.7 APG",
            "Steve Nash": "Steve Nash, Retired, Point Guard, 6'3\", 2-time MVP, 0-time Champion, 14.3 PPG, 3.0 RPG, 8.5 APG"
        }

        self.openai_api_key = "INSERT API KEY HERE"
        self.poisoned_position = poisoned_position

        self.chosen_player = random.choice(list(self.players.keys()))

        self.llama_llm = LlamaOpenAI(model="gpt-4o-mini", api_key=self.openai_api_key)
        self.openai_client = OpenAIAPI(api_key=self.openai_api_key)

        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.facts = []
    
    def generate_fact(self):
        is_poisoned = (len(self.facts) == self.poisoned_position)

        if is_poisoned == False:
            if len(self.facts) == 0:
                prompt = f"Provide the primary team that {self.chosen_player} played for in the following format: Primarily played for XXX."
            elif len(self.facts) == 1:
                prompt = f"Provide the primary position that {self.chosen_player} played in their career in the following format: Primarily played XX."
            elif len(self.facts) == 2:
                prompt = f"Provide the number of MVPs that {self.chosen_player} played for in their career in the following format: Won X NBA MVPs."
            elif len(self.facts) == 3:
                prompt = f"Provide the average career points per game of {self.chosen_player} in the following format: Averaged XX career points per game."
            elif len(self.facts) == 4:
                prompt = f"Provide the number of championships that {self.chosen_player} won in the following format: Won XX NBA championships."
        
        if is_poisoned == True:
            if len(self.facts) == 0:
                prompt = f"Purposefully provide the INCORRECT primary team that {self.chosen_player} played for in the following format: Primarily played for XXX."
            elif len(self.facts) == 1:
                prompt = f"Purposefully provide the INCORRECT primary position that {self.chosen_player} played in their career in the following format: Primarily played XX."
            elif len(self.facts) == 2:
                prompt = f"Purposefully provide the INCORRECT number of MVPs that {self.chosen_player} played for in their career in the following format: Won X NBA MVPs."
            elif len(self.facts) == 3:
                prompt = f"Purposefully provide the INCORRECT average career points per game (by a wide margin) of {self.chosen_player} in the following format: Averaged XX career points per game."
            elif len(self.facts) == 4:
                prompt = f"Purposefully provide the INCORRECT number of championships that {self.chosen_player} won in the following format: Won XX NBA championships."
        
        prompt += "Do not include the name of the player in your response."
        
        response = self.llama_llm.complete(prompt)
        fact = response.text.strip()
        self.facts.append(fact)
    
    def guess_player(self):
        prompt = f"Use the following hints to guess an active or retired NBA player: {' '.join(self.facts)}. The player must be from the following list: {','.join(list(self.players.keys()))}. Respond ONLY with the first and last name of the NBA player."
        response = self.llama_llm.complete(prompt)
        guessed_player = response.text.strip()
        return guessed_player

    def compute_semantic_similarity(self, guess, actual):
        embeddings = self.similarity_model.encode([guess, actual])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return similarity

    def play_game(self):
        similarity_scores = []
        for i in range(5):
            self.generate_fact()
            guessed_player = self.guess_player()

            prompt = f"Generate a statement for {guessed_player} in the following format: Paul Pierce, Retired, Small Forward, 6'7\", 1-time MVP, 1-time Champion, 20.0 PPG, 6.8 RPG, 3.5 APG"
            response = self.llama_llm.complete(prompt)
            guessed_statistic = response.text.strip()

            similarity_score = self.compute_semantic_similarity(guessed_statistic, self.players[self.chosen_player])
            similarity_scores.append(similarity_score)

        return {
            'chosen_player': self.chosen_player,
            'guessed_player': guessed_player,
            'poisoned_position': self.poisoned_position,
            'similarity_score': similarity_scores[-1]
        }
    
def run_multiple_games(num_games, poisoned_positions):
    results = []
    for position in poisoned_positions:
        position_results = []
        for _ in range(num_games):
            game = PlayerGuessingGame(position)
            game_result = game.play_game()
            position_results.append(game_result)
            
        avg_similarity = np.mean([result['similarity_score'] for result in position_results])
        results.append({
            'poisoned_position': position,
            'average_similarity': avg_similarity,
            'game_results': position_results
        })
        
    return results

if __name__ == "__main__":
    experiment_results = run_multiple_games(num_games=10, poisoned_positions=[-1, 0, 1, 2, 3, 4])
    for result in experiment_results:
        print(f"Poisoned Position {result['poisoned_position']}:")
        print(f"Average Similarity: {result['average_similarity']:.4f}")
        print()