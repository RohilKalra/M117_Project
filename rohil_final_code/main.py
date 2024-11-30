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
        "shorten": {
            "position_effects": {},
            "metrics": {},
        },
        "lengthen": {
            "position_effects": {},
            "metrics": {},
        },
    }

    # Run all experiments for both modes
    shorten_results, lengthen_results = experiment.run_all_experiments(book["summary"])

    # Process shorten results
    book_results["shorten"]["clean_summaries"] = shorten_results["clean"]
    for position in range(experiment.num_agents):
        poisoned_summaries = shorten_results["poisoned"][position]
        similarity_metrics = experiment.calculate_similarity_metrics(
            shorten_results["clean"][-1], poisoned_summaries[-1]
        )
        book_results["shorten"]["position_effects"][position] = {
            "summaries": poisoned_summaries,
            "similarities": similarity_metrics,
        }

    # Process lengthen results
    book_results["lengthen"]["clean_summaries"] = lengthen_results["clean"]
    for position in range(experiment.num_agents):
        poisoned_summaries = lengthen_results["poisoned"][position]
        similarity_metrics = experiment.calculate_similarity_metrics(
            lengthen_results["clean"][-1], poisoned_summaries[-1]
        )
        book_results["lengthen"]["position_effects"][position] = {
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
    parser.add_argument(
        "--num_blend_checks_allowed",
        type=int,
        default=2,
        help="Maximum number of allowed blend tool security checks (default: 2)",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    total_start_time = time.time()
    print(
        f"Starting experiment with {args.num_books} books, {args.num_agents} agents, "
        f"and {args.num_blend_checks_allowed} security checks..."
    )

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = "output"
    run_output_dir = os.path.join(
        base_output_dir, f"run_{timestamp}_blendchecks_{args.num_blend_checks_allowed}"
    )
    os.makedirs(run_output_dir, exist_ok=True)

    # Load API key
    print("Loading API key...")
    with open("../secrets.yaml", "r") as stream:
        secrets = yaml.safe_load(stream)
        api_key = secrets["openai_key"]

    # Initialize experiment with command line arguments
    print("Initializing experiment...")
    experiment = PoisonPropagationExperiment(
        api_key,
        num_agents=args.num_agents,
        num_blend_checks_allowed=args.num_blend_checks_allowed,
    )

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
        futures = [executor.submit(process_book, args) for args in process_args]

        for future in futures:
            title, book_results = future.result()
            results[title] = book_results

    # Analyze results for both modes
    print("\nAnalyzing results...")
    analysis = {
        "shorten": experiment.analyze_results(
            {title: data["shorten"] for title, data in results.items()}
        ),
        "lengthen": experiment.analyze_results(
            {title: data["lengthen"] for title, data in results.items()}
        ),
    }

    # Create visualizations for both modes
    print("Creating visualizations...")
    for mode in ["shorten", "lengthen"]:
        create_visualizations(
            analysis[mode],
            timestamp,
            run_output_dir,
            mode=mode,
            filename=f"similarity_analysis_{mode}_{timestamp}.png",
        )

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
            "num_blend_checks_allowed": args.num_blend_checks_allowed,
        },
        "blend_activations": experiment.blend_activations,
    }

    with open(
        os.path.join(run_output_dir, "poison_propagation_results.json"), "w"
    ) as f:
        json.dump(output, f, indent=2)

    # ... (keep all code the same until the analysis summary writing section)

    # Save summary analysis to separate file
    print("Saving analysis summary...")
    with open(os.path.join(run_output_dir, "analysis_summary.txt"), "w") as f:
        f.write("Analysis of Poison Propagation\n")
        f.write("===========================\n\n")
        f.write(f"Experiment Parameters:\n")
        f.write(f"Number of books analyzed: {args.num_books}\n")
        f.write(f"Number of agents: {args.num_agents}\n")
        f.write(
            f"Maximum number of blend tool checks allowed: {args.num_blend_checks_allowed}\n"
        )

        for mode in ["shorten", "lengthen"]:
            f.write(f"\n{mode.capitalize()} Mode Analysis:\n")
            f.write("--------------------\n")
            f.write(
                f"Total API calls to agent.summarize(): {experiment.summarize_calls[mode]}\n"
            )

            # Calculate average blend activations
            clean_activations = experiment.blend_activations[mode]["clean"]
            poisoned_activations = experiment.blend_activations[mode]["poisoned"]
            avg_poisoned_activations = (
                sum(poisoned_activations.values()) / len(poisoned_activations)
                if poisoned_activations
                else 0
            )

            f.write(f"Clean run blend activations: {clean_activations}\n")
            f.write(
                f"Average blend activations per poison position: {avg_poisoned_activations:.2f}\n"
            )

            # Calculate theoretical minimum calls for this mode
            min_calls_per_book = args.num_agents + sum(range(args.num_agents))
            total_min_calls = min_calls_per_book * args.num_books
            f.write(f"Theoretical minimum calls needed: {total_min_calls}\n")
            f.write(
                f"Calls saved by optimization: {total_min_calls - experiment.summarize_calls[mode]}\n\n"
            )

            f.write("Average similarity by poison position:\n")
            for position, similarity in analysis[mode][
                "avg_similarity_by_position"
            ].items():
                f.write(f"Position {position}: {similarity:.4f}\n")

        f.write(f"\nTotal execution time: {total_time:.2f} seconds\n")
        f.write("\nPer-book processing times:\n")
        for title, book_data in results.items():
            f.write(f"{title}: {book_data['processing_time']:.2f} seconds\n")

    # ... (keep rest of the code the same)

    print(f"\nExperiment complete! Results saved in '{run_output_dir}':")
    print(f"- poison_propagation_results.json")
    print(f"- analysis_summary.txt")
    print(f"- similarity_analysis_shorten_{timestamp}.png")
    print(f"- similarity_analysis_lengthen_{timestamp}.png")
    print(f"Total execution time: {total_time:.2f} seconds")

    # Print blend tool statistics
    experiment.print_blend_statistics()


if __name__ == "__main__":
    main()
