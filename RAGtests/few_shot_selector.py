"""
Few-shot example selector with multiple strategies.
Focuses on finding optimal example sets for few-shot prompting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class FewShotSelector:
    """Selects optimal few-shot examples from training data."""
    
    def __init__(self, train_data_path: str, category: str, random_seed: int = 42):
        """
        Initialize selector with training data.
        
        Args:
            train_data_path: Path to training data CSV
            category: Category column name (e.g., "1_D_M")
            random_seed: Random seed for reproducibility
        """
        self.category = category
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Load training data
        self.train_df = pd.read_csv(train_data_path)
        
        # Get valid training examples (non-null ratings)
        self.valid_data = self.train_df[
            self.train_df[category].notna()
        ].copy()
        
        # Get unique rating classes
        self.classes = sorted(self.valid_data[category].unique())
        
        print(f"✓ Loaded {len(self.valid_data)} training examples")
        print(f"  Classes: {self.classes}")
        print(f"  Distribution: {dict(self.valid_data[category].value_counts().sort_index())}")
    
    def select_examples(
        self, 
        n_shots: int, 
        strategy: str = "balanced"
    ) -> List[Dict]:
        """
        Select n_shots examples using specified strategy.
        
        Args:
            n_shots: Number of examples to select
            strategy: Selection strategy name
            
        Returns:
            List of example dictionaries with 'text' and 'rating'
        """
        if strategy == "random":
            return self._select_random(n_shots)
        elif strategy == "balanced":
            return self._select_balanced(n_shots)
        elif strategy == "stratified":
            return self._select_stratified(n_shots)
        elif strategy == "diverse":
            return self._select_diverse(n_shots)
        elif strategy == "representative":
            return self._select_representative(n_shots)
        elif strategy == "boundary":
            return self._select_boundary(n_shots)
        elif strategy == "hard":
            return self._select_hard(n_shots)
        elif strategy == "easy":
            return self._select_easy(n_shots)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _select_random(self, n_shots: int) -> List[Dict]:
        """Random selection from training set."""
        samples = self.valid_data.sample(n=min(n_shots, len(self.valid_data)))
        return self._format_examples(samples)
    
    def _select_balanced(self, n_shots: int) -> List[Dict]:
        """Equal number of examples from each rating class."""
        per_class = max(1, n_shots // len(self.classes))
        examples = []
        
        for rating in self.classes:
            class_data = self.valid_data[self.valid_data[self.category] == rating]
            n_samples = min(per_class, len(class_data))
            samples = class_data.sample(n=n_samples)
            examples.extend(self._format_examples(samples))
        
        # If we need more examples, add randomly
        if len(examples) < n_shots:
            remaining = n_shots - len(examples)
            additional = self.valid_data.sample(n=remaining)
            examples.extend(self._format_examples(additional))
        
        return examples[:n_shots]
    
    def _select_stratified(self, n_shots: int) -> List[Dict]:
        """Stratified sampling maintaining rating distribution."""
        # Calculate proportions
        value_counts = self.valid_data[self.category].value_counts()
        proportions = value_counts / len(self.valid_data)
        
        examples = []
        for rating in self.classes:
            n_samples = max(1, int(n_shots * proportions[rating]))
            class_data = self.valid_data[self.valid_data[self.category] == rating]
            n_samples = min(n_samples, len(class_data))
            samples = class_data.sample(n=n_samples)
            examples.extend(self._format_examples(samples))
        
        return examples[:n_shots]
    
    def _select_diverse(self, n_shots: int) -> List[Dict]:
        """Select diverse examples (varying text length and complexity)."""
        # Add text length as diversity metric
        df = self.valid_data.copy()
        df['text_length'] = df['free_text_answer'].str.len()
        df['word_count'] = df['free_text_answer'].str.split().str.len()
        
        # Sort by length and select evenly spaced examples
        df_sorted = df.sort_values('text_length')
        indices = np.linspace(0, len(df_sorted) - 1, n_shots, dtype=int)
        samples = df_sorted.iloc[indices]
        
        return self._format_examples(samples)
    
    def _select_representative(self, n_shots: int) -> List[Dict]:
        """Select most representative (typical) examples per class."""
        # Use median text length as representativeness proxy
        examples = []
        per_class = max(1, n_shots // len(self.classes))
        
        for rating in self.classes:
            class_data = self.valid_data[self.valid_data[self.category] == rating].copy()
            class_data['text_length'] = class_data['free_text_answer'].str.len()
            median_length = class_data['text_length'].median()
            
            # Find examples closest to median length
            class_data['distance_to_median'] = abs(class_data['text_length'] - median_length)
            samples = class_data.nsmallest(per_class, 'distance_to_median')
            examples.extend(self._format_examples(samples))
        
        return examples[:n_shots]
    
    def _select_boundary(self, n_shots: int) -> List[Dict]:
        """Select examples near class boundaries (ambiguous cases)."""
        # For ordinal classification, boundary cases are between adjacent ratings
        # Use text length variance as boundary proxy (more varied = more ambiguous)
        examples = []
        per_class = max(1, n_shots // len(self.classes))
        
        for rating in self.classes:
            class_data = self.valid_data[self.valid_data[self.category] == rating].copy()
            # Select examples with medium text length (potentially ambiguous)
            class_data['text_length'] = class_data['free_text_answer'].str.len()
            median_length = class_data['text_length'].median()
            
            # Select examples in middle quartiles (avoid extremes)
            q25 = class_data['text_length'].quantile(0.25)
            q75 = class_data['text_length'].quantile(0.75)
            boundary_samples = class_data[
                (class_data['text_length'] >= q25) & 
                (class_data['text_length'] <= q75)
            ]
            
            if len(boundary_samples) >= per_class:
                samples = boundary_samples.sample(n=per_class)
            else:
                samples = class_data.sample(n=min(per_class, len(class_data)))
            
            examples.extend(self._format_examples(samples))
        
        return examples[:n_shots]
    
    def _select_hard(self, n_shots: int) -> List[Dict]:
        """Select hard examples (challenging to classify)."""
        # Proxy: longer texts or texts with unusual characteristics
        df = self.valid_data.copy()
        df['text_length'] = df['free_text_answer'].str.len()
        
        # Select longer, more complex examples
        samples = df.nlargest(n_shots, 'text_length')
        return self._format_examples(samples)
    
    def _select_easy(self, n_shots: int) -> List[Dict]:
        """Select easy examples (clear, prototypical cases)."""
        # Proxy: medium-length texts with common characteristics
        df = self.valid_data.copy()
        df['text_length'] = df['free_text_answer'].str.len()
        
        # Select medium-length examples (easier to understand)
        median_length = df['text_length'].median()
        df['distance_to_median'] = abs(df['text_length'] - median_length)
        samples = df.nsmallest(n_shots, 'distance_to_median')
        
        return self._format_examples(samples)
    
    def _format_examples(self, samples: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame samples to example dictionaries."""
        examples = []
        for _, row in samples.iterrows():
            examples.append({
                'text': row['free_text_answer'],
                'rating': int(row[self.category])
            })
        return examples
    
    def get_example_statistics(self, examples: List[Dict]) -> Dict:
        """Get statistics about selected examples."""
        texts = [ex['text'] for ex in examples]
        ratings = [ex['rating'] for ex in examples]
        
        return {
            'n_examples': len(examples),
            'rating_distribution': {r: ratings.count(r) for r in set(ratings)},
            'avg_text_length': np.mean([len(t) for t in texts]),
            'avg_word_count': np.mean([len(t.split()) for t in texts]),
            'min_length': min([len(t) for t in texts]),
            'max_length': max([len(t) for t in texts])
        }
