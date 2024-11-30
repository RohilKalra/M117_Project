# book_summary_loader.py
import pandas as pd
from typing import List, Dict


class BookSummaryLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_summaries(self) -> List[Dict]:
        """Load book summaries from TSV file"""
        df = pd.read_csv(
            self.file_path,
            sep="\t",
            names=[
                "wiki_id",
                "freebase_id",
                "title",
                "author",
                "pub_date",
                "genres",
                "summary",
            ],
        )

        summaries = []
        for _, row in df.iterrows():
            summaries.append(
                {
                    "title": row["title"],
                    "author": row["author"],
                    "summary": row["summary"],
                }
            )
        return summaries
