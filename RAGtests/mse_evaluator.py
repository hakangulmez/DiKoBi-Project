"""
MSE-focused evaluation utilities for few-shot experiments.
Prioritizes Mean Squared Error with detailed error analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.metrics import confusion_matrix, cohen_kappa_score


class MSEEvaluator:
    """Evaluator focusing on MSE and error analysis."""
    
    def __init__(self, y_true: List[int], y_pred: List[int]):
        """
        Initialize evaluator with predictions.
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.errors = self.y_pred - self.y_true
        self.squared_errors = self.errors ** 2
    
    def calculate_all_metrics(self) -> Dict:
        """Calculate comprehensive metrics with MSE focus."""
        metrics = {
            # Primary metric
            'mse': mean_squared_error(self.y_true, self.y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            
            # Secondary metrics
            'mae': mean_absolute_error(self.y_true, self.y_pred),
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            
            # Error distribution
            'mean_error': float(np.mean(self.errors)),
            'std_error': float(np.std(self.errors)),
            'max_overestimation': float(np.max(self.errors)),
            'max_underestimation': float(np.min(self.errors)),
            
            # Detailed analysis
            'n_samples': len(self.y_true),
            'n_correct': int(np.sum(self.y_true == self.y_pred)),
            'n_off_by_1': int(np.sum(np.abs(self.errors) == 1)),
            'n_off_by_2': int(np.sum(np.abs(self.errors) == 2)),
            'n_off_by_3_plus': int(np.sum(np.abs(self.errors) >= 3)),
        }
        
        # Add QWK if possible
        try:
            metrics['qwk'] = cohen_kappa_score(
                self.y_true, 
                self.y_pred, 
                weights='quadratic'
            )
        except:
            metrics['qwk'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(
            self.y_true, 
            self.y_pred
        ).tolist()
        
        return metrics
    
    def get_error_breakdown(self) -> Dict:
        """Detailed breakdown of errors."""
        breakdown = {
            'perfect': int(np.sum(self.errors == 0)),
            'overestimated_by_1': int(np.sum(self.errors == 1)),
            'overestimated_by_2': int(np.sum(self.errors == 2)),
            'overestimated_by_3+': int(np.sum(self.errors >= 3)),
            'underestimated_by_1': int(np.sum(self.errors == -1)),
            'underestimated_by_2': int(np.sum(self.errors == -2)),
            'underestimated_by_3+': int(np.sum(self.errors <= -3)),
        }
        
        # Calculate percentages
        total = len(self.errors)
        breakdown['percentages'] = {
            k: f"{(v/total)*100:.1f}%" for k, v in breakdown.items()
        }
        
        return breakdown
    
    def get_worst_predictions(self, n: int = 10) -> List[Tuple[int, int, int]]:
        """
        Get predictions with highest squared errors.
        
        Args:
            n: Number of worst predictions to return
            
        Returns:
            List of (index, true_rating, predicted_rating) tuples
        """
        worst_indices = np.argsort(self.squared_errors)[-n:][::-1]
        
        worst = []
        for idx in worst_indices:
            worst.append((
                int(idx),
                int(self.y_true[idx]),
                int(self.y_pred[idx])
            ))
        
        return worst
    
    def get_class_specific_mse(self) -> Dict[int, float]:
        """Calculate MSE for each true rating class."""
        class_mse = {}
        
        for rating in np.unique(self.y_true):
            mask = self.y_true == rating
            if np.sum(mask) > 0:
                class_mse[int(rating)] = float(
                    mean_squared_error(
                        self.y_true[mask], 
                        self.y_pred[mask]
                    )
                )
        
        return class_mse
    
    def print_summary(self, category: str = ""):
        """Print comprehensive evaluation summary."""
        metrics = self.calculate_all_metrics()
        breakdown = self.get_error_breakdown()
        class_mse = self.get_class_specific_mse()
        
        print(f"\n{'='*70}")
        print(f"MSE EVALUATION SUMMARY{' - ' + category if category else ''}")
        print(f"{'='*70}")
        
        print(f"\n📊 PRIMARY METRICS (MSE Focus)")
        print(f"  MSE:        {metrics['mse']:.4f}  ⭐ (Lower is better)")
        print(f"  RMSE:       {metrics['rmse']:.4f}")
        print(f"  MAE:        {metrics['mae']:.4f}")
        print(f"  Accuracy:   {metrics['accuracy']:.1%}")
        if metrics['qwk'] is not None:
            print(f"  QWK:        {metrics['qwk']:.4f}")
        
        print(f"\n📈 ERROR STATISTICS")
        print(f"  Mean Error:     {metrics['mean_error']:+.3f}")
        print(f"  Std Error:      {metrics['std_error']:.3f}")
        print(f"  Max Over:       {metrics['max_overestimation']:+d}")
        print(f"  Max Under:      {metrics['max_underestimation']:+d}")
        
        print(f"\n🎯 ERROR BREAKDOWN")
        print(f"  Perfect:              {breakdown['perfect']:3d} ({breakdown['percentages']['perfect']})")
        print(f"  Off by 1:             {metrics['n_off_by_1']:3d}")
        print(f"    - Overestimated:    {breakdown['overestimated_by_1']:3d} ({breakdown['percentages']['overestimated_by_1']})")
        print(f"    - Underestimated:   {breakdown['underestimated_by_1']:3d} ({breakdown['percentages']['underestimated_by_1']})")
        print(f"  Off by 2:             {metrics['n_off_by_2']:3d}")
        print(f"    - Overestimated:    {breakdown['overestimated_by_2']:3d} ({breakdown['percentages']['overestimated_by_2']})")
        print(f"    - Underestimated:   {breakdown['underestimated_by_2']:3d} ({breakdown['percentages']['underestimated_by_2']})")
        print(f"  Off by 3+:            {metrics['n_off_by_3_plus']:3d}")
        
        print(f"\n📋 CLASS-SPECIFIC MSE")
        for rating, mse in sorted(class_mse.items()):
            n_samples = int(np.sum(self.y_true == rating))
            print(f"  Rating {rating}: MSE={mse:.4f} (n={n_samples})")
        
        print(f"\n🔢 CONFUSION MATRIX")
        cm = np.array(metrics['confusion_matrix'])
        ratings = sorted(np.unique(np.concatenate([self.y_true, self.y_pred])))
        
        # Print header
        print(f"  True\\Pred  ", end="")
        for r in ratings:
            print(f"{r:4d}", end="")
        print()
        
        # Print rows
        for i, true_rating in enumerate(ratings):
            print(f"    {true_rating:4d}     ", end="")
            for j, pred_rating in enumerate(ratings):
                if i < len(cm) and j < len(cm[i]):
                    print(f"{cm[i][j]:4d}", end="")
                else:
                    print(f"   0", end="")
            print()
        
        print(f"\n{'='*70}\n")
    
    def compare_results(self, other: 'MSEEvaluator', label1: str = "A", label2: str = "B"):
        """
        Compare two sets of results.
        
        Args:
            other: Another MSEEvaluator instance
            label1: Label for this evaluator
            label2: Label for other evaluator
        """
        metrics1 = self.calculate_all_metrics()
        metrics2 = other.calculate_all_metrics()
        
        print(f"\n{'='*70}")
        print(f"COMPARISON: {label1} vs {label2}")
        print(f"{'='*70}")
        
        print(f"\n{'Metric':<15} {label1:>15} {label2:>15} {'Difference':>15}")
        print(f"{'-'*70}")
        
        # Compare key metrics
        for metric in ['mse', 'rmse', 'mae', 'accuracy']:
            val1 = metrics1[metric]
            val2 = metrics2[metric]
            diff = val2 - val1
            
            if metric == 'accuracy':
                print(f"{metric.upper():<15} {val1:>14.1%} {val2:>14.1%} {diff:>+14.1%}")
            else:
                print(f"{metric.upper():<15} {val1:>15.4f} {val2:>15.4f} {diff:>+15.4f}")
        
        # Determine winner
        if metrics2['mse'] < metrics1['mse']:
            winner = label2
            improvement = ((metrics1['mse'] - metrics2['mse']) / metrics1['mse']) * 100
        else:
            winner = label1
            improvement = ((metrics2['mse'] - metrics1['mse']) / metrics2['mse']) * 100
        
        print(f"\n⭐ WINNER: {winner} (MSE improved by {improvement:.1f}%)")
        print(f"{'='*70}\n")


def compare_multiple_results(
    results: Dict[str, Tuple[List[int], List[int]]],
    sort_by: str = 'mse'
) -> pd.DataFrame:
    """
    Compare multiple experiment results.
    
    Args:
        results: Dictionary mapping labels to (y_true, y_pred) tuples
        sort_by: Metric to sort by ('mse', 'mae', 'accuracy')
        
    Returns:
        DataFrame with comparison results
    """
    comparison = []
    
    for label, (y_true, y_pred) in results.items():
        evaluator = MSEEvaluator(y_true, y_pred)
        metrics = evaluator.calculate_all_metrics()
        
        comparison.append({
            'Experiment': label,
            'MSE': metrics['mse'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'Accuracy': metrics['accuracy'],
            'QWK': metrics.get('qwk', None),
            'N_Samples': metrics['n_samples']
        })
    
    df = pd.DataFrame(comparison)
    
    # Sort by specified metric
    ascending = sort_by != 'accuracy' and sort_by != 'qwk'
    df = df.sort_values(sort_by.upper(), ascending=ascending)
    
    # Add rank
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    return df
