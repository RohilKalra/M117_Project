# .PHONY: run clean

# Run the experiment
run:
	python main.py --num_books 2 --num_agents 3

# Clean up generated files and results
clean:
	rm -rf __pycache__/
	rm -f poison_propagation_results.json
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete