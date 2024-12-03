import openai
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


with open("secrets.yaml", "r") as stream:
    secrets = yaml.safe_load(stream)
    openai.api_key = secrets["openai_key"]


recipe_database = {
    "Spaghetti Bolognese": {
        "ingredients": [
            "spaghetti", "ground beef", "tomato", "onion", "garlic", "carrot", "celery", "parmesan", "olive oil", "salt", "pepper"
        ],
        "steps": [
            "1. Heat olive oil in a pan over medium heat.",
            "2. Add chopped onions and garlic to the pan, sauté until translucent.",
            "3. Add ground beef to the pan and cook until browned.",
            "4. Add diced tomatoes, carrots, and celery, then simmer for 15 minutes.",
            "5. Cook spaghetti according to the package instructions in salted water.",
            "6. Serve the sauce over the cooked spaghetti and garnish with parmesan."
        ]
    },
    "Chicken Alfredo": {
        "ingredients": [
            "fettuccine", "chicken breast", "cream", "butter", "garlic", "parmesan", "salt", "pepper", "olive oil"
        ],
        "steps": [
            "1. Cook fettuccine according to the package instructions.",
            "2. Heat olive oil in a pan, then cook the chicken breast until browned and fully cooked.",
            "3. Remove chicken from the pan and set aside.",
            "4. In the same pan, melt butter and sauté garlic until fragrant.",
            "5. Add cream and bring to a simmer, then stir in parmesan cheese.",
            "6. Slice the chicken and toss with the fettuccine and sauce. Serve immediately."
        ]
    },
    "Vegetable Stir-Fry": {
        "ingredients": [
            "broccoli", "carrot", "bell pepper", "onion", "garlic", "soy sauce", "sesame oil", "ginger", "tofu", "sesame seeds"
        ],
        "steps": [
            "1. Cut all vegetables into bite-sized pieces.",
            "2. Heat sesame oil in a large pan, then sauté garlic and ginger until fragrant.",
            "3. Add the vegetables to the pan and stir-fry for 5-7 minutes until tender but crisp.",
            "4. Add tofu and soy sauce, then cook for an additional 3-5 minutes.",
            "5. Garnish with sesame seeds and serve.",
            "6. Enjoy your vegetable stir-fry!"
        ]
    },
    "Beef Tacos": {
        "ingredients": [
            "ground beef", "taco seasoning", "taco shells", "lettuce", "tomato", "cheese", "sour cream", "onion", "jalapeños"
        ],
        "steps": [
            "1. Cook ground beef in a pan, adding taco seasoning according to package instructions.",
            "2. Warm the taco shells in the oven.",
            "3. Chop lettuce, tomato, and onion, and slice the jalapeños.",
            "4. Assemble the tacos by filling each shell with seasoned beef and topping with lettuce, tomato, cheese, sour cream, and jalapeños.",
            "5. Serve the tacos immediately.",
            "6. Enjoy your delicious beef tacos!"
        ]
    },
    "Margherita Pizza": {
        "ingredients": [
            "pizza dough", "tomato sauce", "mozzarella cheese", "basil", "olive oil", "salt"
        ],
        "steps": [
            "1. Preheat the oven to 475°F (245°C).",
            "2. Roll out the pizza dough on a floured surface.",
            "3. Spread tomato sauce evenly on the dough.",
            "4. Top with mozzarella cheese and fresh basil leaves.",
            "5. Drizzle with olive oil and sprinkle with a pinch of salt.",
            "6. Bake for 10-12 minutes until the crust is golden and cheese is bubbly."
        ]
    },
    "Grilled Cheese Sandwich": {
        "ingredients": [
            "bread", "butter", "cheddar cheese", "tomato"
        ],
        "steps": [
            "1. Butter one side of each slice of bread.",
            "2. Place a slice of cheese between the unbuttered sides of the bread.",
            "3. Heat a pan over medium heat and grill the sandwich until golden brown on both sides.",
            "4. Serve with a side of sliced tomatoes.",
            "5. Enjoy your grilled cheese sandwich!",
            "6. For extra flavor, add some tomato soup!"
        ]
    },
    "Caesar Salad": {
        "ingredients": [
            "romaine lettuce", "croutons", "parmesan", "Caesar dressing", "garlic", "lemon"
        ],
        "steps": [
            "1. Tear the romaine lettuce into bite-sized pieces.",
            "2. Toss the lettuce with Caesar dressing.",
            "3. Add croutons and parmesan cheese.",
            "4. Squeeze a lemon over the salad.",
            "5. Serve immediately.",
            "6. Enjoy your fresh Caesar salad!"
        ]
    },
    "Pancakes": {
        "ingredients": [
            "flour", "baking powder", "milk", "egg", "butter", "sugar", "vanilla", "maple syrup"
        ],
        "steps": [
            "1. In a bowl, mix flour, baking powder, and sugar.",
            "2. Add milk, egg, and melted butter, then stir until smooth.",
            "3. Heat a pan over medium heat and pour batter onto the pan in small circles.",
            "4. Cook until bubbles appear on the surface, then flip and cook until golden brown.",
            "5. Serve with maple syrup and butter.",
            "6. Enjoy your delicious pancakes!"
        ]
    },
    "Chicken Caesar Wrap": {
        "ingredients": [
            "chicken breast", "romaine lettuce", "Caesar dressing", "tortilla wraps", "parmesan", "croutons"
        ],
        "steps": [
            "1. Cook the chicken breast and slice it into strips.",
            "2. Toss the chicken with romaine lettuce and Caesar dressing.",
            "3. Sprinkle parmesan cheese and croutons over the salad.",
            "4. Place the mixture in a tortilla wrap and roll it up tightly.",
            "5. Slice and serve.",
            "6. Enjoy your Chicken Caesar Wrap!"
        ]
    },
    "Vegetable Soup": {
        "ingredients": [
            "carrot", "potato", "onion", "celery", "garlic", "tomato", "vegetable broth", "salt", "pepper", "herbs"
        ],
        "steps": [
            "1. Chop all vegetables into bite-sized pieces.",
            "2. Heat oil in a pot and sauté onion, garlic, and celery until softened.",
            "3. Add the remaining vegetables and vegetable broth.",
            "4. Bring to a boil, then reduce to a simmer and cook for 20-30 minutes.",
            "5. Season with salt, pepper, and herbs to taste before serving.",
            "6. Serve hot and enjoy your vegetable soup!"
        ]
    }
}


# InstructionAgent generates the current instruction step based on the previous step
class InstructionAgent:
    def __init__(self):
        self.step_counter = 1
        self.poison_step = 2  # Randomly select a step to poison
        print("Poisoned: " + str(self.poison_step))
        print("----------")

    def generate_next_step(self, steps_so_far: list[str], meal: str) -> str:
        self.step_counter += 1  # Calculate the step number
        is_poison_step = self.step_counter == self.poison_step

        steps_text = "\n".join(steps_so_far)  # Combine all prior steps

        if is_poison_step:
            prompt = f"""
            Given the recipe for {meal}, here are the steps so far:

            {steps_text}

            Now, create the next step. For this specific step, you must include an outrageous or disastrous action that will alter the entire recipe irreparably. Keep the tone subtle yet impactful.

            Follow this format:
            {self.step_counter}. Instruction Step
            """
        else:
            prompt = f"""
            Given the recipe for {meal}, here are the steps so far:

            {steps_text}

            Based on the instructions above, what is the next step? Ensure it flows logically while building on all previous steps.

            Follow this format:
            {self.step_counter}. Instruction Step
            """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000,
        )

        generated_step = response['choices'][0]['message']['content'].strip()

        return generated_step


# RecipeAgent orchestrates the sequence of LLM calls
class RecipeAgent:
    def __init__(self):
        self.llm_agent = InstructionAgent()


    def generate_recipe(self, meal: str, first_step: str) -> list[str]:
        """Generates the full recipe by calling LLM agents in sequence."""
        steps = []

        # Start with the first instruction
        steps.append(first_step)
        
        # Generate subsequent steps by calling LLM for each one
        for i in range(1, 6):  # 5 more steps
            previous_step = steps[-1]  # Get the last step
            next_step = self.llm_agent.generate_next_step(previous_step, meal)
            steps.append(next_step)

        return steps


def jaccard_similarity(step1: str, step2: str) -> float:
    """
    Calculate Jaccard Similarity between two strings.
    Tokenizes the strings into sets of words and computes the similarity.
    Returns a score between 0 and 1.
    """
    set1 = set(step1.split())  # Split words in step1
    set2 = set(step2.split())  # Split words in step2
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def semantic_similarity(step1: str, step2: str) -> float:
    """
    Calculate semantic similarity using sentence embeddings.
    Returns a score between 0 and 1.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([step1, step2])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])


def comprehensive_recipe_similarity(control_step: str, generated_step: str) -> float:
    """
    Combine multiple similarity metrics for a comprehensive similarity score.
    Returns a score between 0 and 1.
    """
    jaccard_sim = jaccard_similarity(control_step, generated_step)
    semantic_sim = semantic_similarity(control_step, generated_step)
    
    # Weighted average (can adjust weights if needed)
    return (0.2 * jaccard_sim + 0.8 * semantic_sim)


def compare_recipes_comprehensive(control_steps: list[str], generated_steps: list[str]) -> float:
    """
    Compare two recipes using a comprehensive similarity approach.
    Returns the average similarity score across all steps.
    """
    similarities = []
    for control_step, generated_step in zip(control_steps, generated_steps):
        similarity = comprehensive_recipe_similarity(control_step, generated_step)
        similarities.append(similarity)
        print(f"Control Step: {control_step}")
        print(f"Generated Step: {generated_step}")
        print(f"Comprehensive Similarity: {similarity:.2f}")
        print("---")
    
    average_similarity = sum(similarities) / len(similarities) if similarities else 0
    print(f"Average Comprehensive Similarity: {average_similarity:.2f}")
    return average_similarity


def main():
    rag_agent = RecipeAgent()

    random_int = random.randint(1, 10)

    meal = "Spaghetti Bolognese"

    generated_steps = rag_agent.generate_recipe(meal, recipe_database[meal]["steps"][0])

    # Output
    for step in generated_steps:
        print(step)
    
    print()
    print("---------- COMPARISON ----------")
    print()

    # Compare with control steps from the database
    control_steps = recipe_database[meal]["steps"]
    compare_recipes_comprehensive(control_steps, generated_steps)


# Run the main function
if __name__ == "__main__":
    main()
