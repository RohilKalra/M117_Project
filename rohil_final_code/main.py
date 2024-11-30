import yaml
import json
from book_summary_loader import BookSummaryLoader
from experiment import PoisonPropagationExperiment
from visualizations import create_visualizations
from datetime import datetime
import time
import os
import argparse
from concurrent.futures import ThreadPoolExecutor


def process_book(args):
    """Process a single book"""
    book, experiment, book_index, total_books = args
    book_start_time = time.time()

    experiment.safe_print(
        f"\nProcessing book {book_index+1}/{total_books}: {book['title']}"
    )

    book_results = {
        "title": book["title"],
        "author": book["author"],
        "position_effects": {},
        "metrics": {},
    }

    # Run all experiments at once
    experiment_results = experiment.run_experiments(book["summary"])
    book_results["clean_summaries"] = experiment_results["clean"]

    # Process results for each position
    for position in range(experiment.num_agents):
        poisoned_summaries = experiment_results["poisoned"][position]

        similarity_metrics = experiment.calculate_similarity_metrics(
            experiment_results["clean"][-1], poisoned_summaries[-1]
        )

        book_results["position_effects"][position] = {
            "summaries": poisoned_summaries,
            "similarities": similarity_metrics,
        }

    book_time = time.time() - book_start_time
    book_results["processing_time"] = book_time
    experiment.safe_print(
        f"Book processing time for {book['title']}: {book_time:.2f} seconds"
    )

    return book["title"], book_results


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run poison propagation experiment on book summaries"
    )
    parser.add_argument(
        "--num_books",
        type=int,
        default=1,
        help="Number of books to analyze (default: 1)",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=6,
        help="Number of agents in the summarization chain (default: 6)",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    total_start_time = time.time()
    print(
        f"Starting experiment with {args.num_books} books and {args.num_agents} agents..."
    )

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = "output"
    run_output_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Load API key
    print("Loading API key...")
    with open("../secrets.yaml", "r") as stream:
        secrets = yaml.safe_load(stream)
        api_key = secrets["openai_key"]

    # Initialize experiment with command line argument
    print("Initializing experiment...")
    experiment = PoisonPropagationExperiment(api_key, num_agents=args.num_agents)

    # Load book summaries
    print("Loading book summaries...")
    loader = BookSummaryLoader("book_summaries.txt")
    book_summaries = loader.load_summaries()[: args.num_books]

    # Prepare arguments for parallel processing
    process_args = [
        (book, experiment, i, len(book_summaries))
        for i, book in enumerate(book_summaries)
    ]

    # Process books in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=min(4, len(book_summaries))) as executor:
        # Submit all tasks and get futures
        futures = [executor.submit(process_book, args) for args in process_args]

        # Collect results as they complete
        for future in futures:
            title, book_results = future.result()
            results[title] = book_results

    # Analyze results
    print("\nAnalyzing results...")
    analysis = experiment.analyze_results(results)

    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(analysis, timestamp, run_output_dir)

    # Save detailed results and analysis
    print("Saving detailed results...")
    total_time = time.time() - total_start_time
    output = {
        "experiment_results": results,
        "analysis": analysis,
        "total_execution_time": total_time,
        "experiment_parameters": {
            "num_books": args.num_books,
            "num_agents": args.num_agents,
        },
    }

    with open(
        os.path.join(run_output_dir, "poison_propagation_results.json"), "w"
    ) as f:
        json.dump(output, f, indent=2)

    # Save summary analysis to separate file
    print("Saving analysis summary...")
    with open(os.path.join(run_output_dir, "analysis_summary.txt"), "w") as f:
        f.write("Analysis of Poison Propagation\n")
        f.write("===========================\n\n")
        f.write(f"Experiment Parameters:\n")
        f.write(f"Number of books analyzed: {args.num_books}\n")
        f.write(f"Number of agents: {args.num_agents}\n")
        f.write(
            f"Total API calls to agent.summarize(): {experiment.summarize_calls}\n\n"
        )

        # Calculate theoretical minimum calls
        min_calls_per_book = args.num_agents + sum(range(args.num_agents))
        total_min_calls = min_calls_per_book * args.num_books
        f.write(f"Theoretical minimum calls needed: {total_min_calls}\n")
        f.write(
            f"Calls saved by optimization: {-(total_min_calls - experiment.summarize_calls)}\n\n"
        )

        f.write("Average similarity by poison position:\n")
        for position, similarity in analysis["avg_similarity_by_position"].items():
            f.write(f"Position {position}: {similarity:.4f}\n")
        f.write(f"\nTotal execution time: {total_time:.2f} seconds\n")

        # Add per-book timing
        f.write("\nPer-book processing times:\n")
        for title, book_data in results.items():
            f.write(f"{title}: {book_data['processing_time']:.2f} seconds\n")

    print(f"\nExperiment complete! Results saved in '{run_output_dir}':")
    print(f"- poison_propagation_results.json")
    print(f"- analysis_summary.txt")
    print(f"- similarity_analysis_{timestamp}.png")
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
