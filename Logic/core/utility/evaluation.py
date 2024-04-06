import math
from typing import List

import numpy
import wandb as wandb


class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0

        if len(predicted) == 0:
            return 0.0

        true_positives = sum(1 for p in predicted if p in actual)
        precision = true_positives / len(predicted)

        return precision
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0

        if len(actual) == 0:
            return 0.0

        true_positives = sum(1 for p in predicted if p in actual)
        recall = true_positives / len(actual)

        return recall
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        f1 = 0.0

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)

        return f1
    
    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = 0.0

        if len(predicted) == 0:
            return 0.0

        num_correct = 0
        total_precision = 0.0
        for i, p in enumerate(predicted):
            if p in actual:
                num_correct += 1
                total_precision += num_correct / (i + 1)

        AP = total_precision / len(actual) if len(actual) > 0 else 0.0

        return AP
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        MAP = 0.0

        total_AP = sum(self.calculate_AP([a], [p]) for a, p in zip(actual, predicted))
        MAP = total_AP / len(actual) if len(actual) > 0 else 0.0

        return MAP
    
    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0
        for i, item in enumerate(predicted):
            if item in actual:
                DCG += 1 / math.log2(i + 2)
        return DCG

        return DCG
    
    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0

        DCG = self.calculate_DCG(actual, predicted)
        ideal_DCG = self.calculate_DCG(predicted, actual)
        NDCG = DCG / ideal_DCG if ideal_DCG != 0 else 0.0

        return NDCG
    
    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RR = 0.0

        for i, p in enumerate(predicted):
            if p in actual:
                RR = 1 / (i + 1)
                break

        return RR
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        MRR = []

        for i,p in enumerate(predicted):
            MRR.append(self.cacluate_RR(actual[i], p))

        MRR = numpy.mean(MRR)
        return MRR

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Average Precision: {ap}")
        print(f"Mean Average Precision: {map}")
        print(f"Discounted Cumulative Gain: {dcg}")
        print(f"Normalized Discounted Cumulative Gain: {ndcg}")
        print(f"Reciprocal Rank: {rr}")
        print(f"Mean Reciprocal Rank: {mrr}")

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """

        wandb.log({
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Average Precision": ap,
            "Mean Average Precision": map,
            "Discounted Cumulative Gain": dcg,
            "Normalized Discounted Cumulative Gain": ndcg,
            "Reciprocal Rank": rr,
            "Mean Reciprocal Rank": mrr
        })


    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)



