"""
Prompt template generator for few-shot experiments.
Creates various prompt formats optimized for small models.
"""

from typing import List, Dict
from src.classification.prompts import CATEGORIES


class FewShotPromptGenerator:
    """Generates few-shot prompts with various formatting strategies."""
    
    def __init__(self, category: str):
        """
        Initialize prompt generator.
        
        Args:
            category: Category name (e.g., "1_D_M")
        """
        self.category = category
        self.category_info = CATEGORIES.get(category, {})
        self.category_name = self.category_info.get('name', category)
        self.definition = self.category_info.get('definition', '[Definition not available]')
        self.rating_scale = self.category_info.get('rating_scale', {})
    
    def generate_prompt(
        self, 
        text: str, 
        examples: List[Dict],
        format_type: str = "standard"
    ) -> str:
        """
        Generate prompt with few-shot examples.
        
        Args:
            text: Text to classify
            examples: List of {'text': str, 'rating': int} dictionaries
            format_type: Prompt format type
            
        Returns:
            Complete prompt string
        """
        if format_type == "standard":
            return self._standard_format(text, examples)
        elif format_type == "explicit_scale":
            return self._explicit_scale_format(text, examples)
        elif format_type == "step_by_step":
            return self._step_by_step_format(text, examples)
        elif format_type == "criteria_focused":
            return self._criteria_focused_format(text, examples)
        elif format_type == "comparative":
            return self._comparative_format(text, examples)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def _standard_format(self, text: str, examples: List[Dict]) -> str:
        """Standard few-shot format."""
        prompt = f"""You are an expert in educational assessment following the DiKoBi coding manual.

Category: {self.category_name}

Rating Scale:
"""
        # Add rating scale descriptions
        for rating, description in sorted(self.rating_scale.items()):
            prompt += f"{rating}: {description}\n"
        
        prompt += "\nExamples:\n\n"
        
        # Add few-shot examples
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f'Text: "{example["text"]}"\n'
            prompt += f'Rating: {example["rating"]}\n\n'
        
        # Add test case
        prompt += "Now analyze this teacher response and provide ONLY the rating number:\n\n"
        prompt += f'Text: "{text}"\n\n'
        prompt += "Rating:"
        
        return prompt
    
    def _explicit_scale_format(self, text: str, examples: List[Dict]) -> str:
        """Format with explicit emphasis on rating scale."""
        prompt = f"""Task: Rate teacher responses on diagnostic competency.

Category: {self.category_name}

RATING SCALE (Choose ONLY one number):
"""
        for rating, description in sorted(self.rating_scale.items()):
            prompt += f"  {rating} = {description}\n"
        
        prompt += "\nLEARN FROM THESE EXAMPLES:\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"\n[Example {i}] Rating: {example['rating']}\n"
            prompt += f'"{example["text"]}"\n'
        
        prompt += f"\nNOW RATE THIS RESPONSE (output only the number):\n"
        prompt += f'"{text}"\n\n'
        prompt += "Rating:"
        
        return prompt
    
    def _step_by_step_format(self, text: str, examples: List[Dict]) -> str:
        """Chain-of-thought format for reasoning."""
        prompt = f"""Analyze teacher responses step-by-step.

Category: {self.category_name}

Rating Scale:
"""
        for rating, description in sorted(self.rating_scale.items()):
            prompt += f"{rating}: {description}\n"
        
        prompt += "\nExamples of step-by-step analysis:\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f'Response: "{example["text"]}"\n'
            prompt += f'Analysis: This response shows characteristics of rating {example["rating"]}.\n'
            prompt += f'Rating: {example["rating"]}\n'
        
        prompt += f"\nNow analyze this response:\n"
        prompt += f'Response: "{text}"\n'
        prompt += "Provide ONLY the rating number:\n\n"
        prompt += "Rating:"
        
        return prompt
    
    def _criteria_focused_format(self, text: str, examples: List[Dict]) -> str:
        """Format highlighting evaluation criteria."""
        prompt = f"""Evaluate based on specific criteria.

Category: {self.category_name}
Definition: {self.definition}

Evaluation Criteria:
"""
        for rating, description in sorted(self.rating_scale.items()):
            prompt += f"Rating {rating}: {description}\n"
        
        prompt += "\nReference Examples:\n"
        
        for i, example in enumerate(examples, 1):
            rating = example['rating']
            criteria = self.rating_scale.get(rating, 'See scale above')
            prompt += f"\n{i}. [Rating {rating}] - {criteria}\n"
            prompt += f'   "{example["text"]}"\n'
        
        prompt += f"\nEvaluate this response using the same criteria:\n"
        prompt += f'"{text}"\n\n'
        prompt += "Rating (number only):"
        
        return prompt
    
    def _comparative_format(self, text: str, examples: List[Dict]) -> str:
        """Format encouraging comparison to examples."""
        prompt = f"""Compare this response to the examples and assign a rating.

Category: {self.category_name}

Rating Scale:
"""
        for rating, description in sorted(self.rating_scale.items()):
            prompt += f"{rating}: {description}\n"
        
        prompt += "\nReference Examples by Rating:\n"
        
        # Group examples by rating
        by_rating = {}
        for example in examples:
            rating = example['rating']
            if rating not in by_rating:
                by_rating[rating] = []
            by_rating[rating].append(example['text'])
        
        for rating in sorted(by_rating.keys()):
            prompt += f"\nRating {rating} examples:\n"
            for i, text_example in enumerate(by_rating[rating], 1):
                prompt += f'  {i}. "{text_example}"\n'
        
        prompt += f"\nCompare this response to the examples above:\n"
        prompt += f'"{text}"\n\n'
        prompt += "Most similar to rating (number only):"
        
        return prompt
    
    def get_available_formats(self) -> List[str]:
        """Return list of available prompt formats."""
        return [
            "standard",
            "explicit_scale",
            "step_by_step",
            "criteria_focused",
            "comparative"
        ]
