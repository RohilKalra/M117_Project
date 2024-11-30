1) Go to the google doc to access my API Key. Fill it into to secrets.yaml like
openai_key: "insert_key_here"

2) For Midterm code, Enter this in terminal

```
pip3 install -r requirements.txt
cd midterm_code
python3 rohil_experiment2.py
```



3. FINAL CODE:

Environment Setup
------------------
First, create and activate the conda environment:

```
conda env create -f environment.yml
conda activate llm_exp
```

Running the Experiment
-----------------------
The program can be run with default parameters:

```
python main.py
```

Or with custom parameters:

```
python main.py --num_books 2 --num_agents 4 --num_blend_checks_allowed 1
```

Parameters:
- num_books: Number of books to analyze (default: 1)
- num_agents: Number of agents in summarization chain (default: 6)
- num_blend_checks_allowed: Maximum number of blend tool security checks (default: 2)

4. Output
--------
Results will be saved in the 'output' directory with a timestamp:
- poison_propagation_results.json: Detailed experimental results
- analysis_summary.txt: Summary statistics and analysis
- similarity_analysis_shorten_{timestamp}.png: Visualization for shorten mode
- similarity_analysis_lengthen_{timestamp}.png: Visualization for lengthen mo