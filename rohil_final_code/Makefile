# .PHONY: run clean

# Run a 3 minute test
run_small_test:
	python main.py --num_books 2 --num_agents 3 --num_blend_checks_allowed 2

# Run a 10 minute test
run_large_test_nbc:
	python main.py --num_books 5 --num_agents 8 --num_blend_checks_allowed 0

run_large_test_wbc:
	python main.py --num_books 5 --num_agents 8 --num_blend_checks_allowed 3


# Clean up generated files and results
clean:
	rm -rf __pycache__/
	rm -f poison_propagation_results.json
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete