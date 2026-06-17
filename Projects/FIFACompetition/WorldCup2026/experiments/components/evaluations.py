import numpy as np
import math
import numpy.typing as npt

def interpret_probabilities(prob):
    """ 
    Function to interpret the returned probabilities by the model with shape (121,)
    
    Args:
        prob (ndarray) - array with probabilities for different match results
        
    Returns:
        scores (str) - human readable version of prediction
    """
    predicted = np.argmax(prob)
    home_score = math.floor(predicted / 11)
    away_score = predicted % 11
    return f"{home_score}-{away_score}"

def score_to_index(score: str):
    """ 
    This function takes string representation of score and converts it to index in class list
    
    Args:
        score (scalar): string representation of score e.g. 1-0
        
    Returns:
        index (scalar): index of that score in class list e.g. 11 for 1-0
    """
    scores = score.split('-')
    assert len(scores) == 2, "Invalid score"
    
    home_score = int(scores[0])
    away_score = int(scores[1])
    
    return (home_score * 11) + away_score

class PredictionRecall:
    def __init__(self, t_p: int, f_p: int, f_n:int, total: int, label: str):
        """ 
        Return object with prediction recall values
        
        Args:
            label (str): Name of the class
            t_p (int): Total true positive for the given label
            f_p (int): Total false positives for given label
            total (int): Total items with given label
            f_n (int): Total false negatives for given label
        """
        self.total = total
        self.t_p = t_p
        self.f_p = f_p
        self.f_n = f_n
        self.label = label
        
    def __repr__(self):
        return f"{self.label} with t_p: {self.t_p}, f_p: {self.f_p}, f_n: {self.f_n} and total: {self.total}"



class Evaluation:
    def accuracy_score(y_pred, y_label):
        """ 
        Return the ratio of correct responses across the entire test set.add
        
        Args:
            y_pred (ndarray): (121, m) array with predictions for each class from the model from the model
            y_label(ndarray): (121, m) array with the actual labels
            
        Returns:
            accuracy (scalar): ratio of correct responses to all responses
        """
        predictions = np.argmax(y_pred, axis=0)
        actual = np.argmax(y_label, axis=0)
        num_correct = np.sum(predictions == actual)
        return num_correct / y_pred.shape[1]
    
    def get_precision_recall(y_pred, y_label) -> list[PredictionRecall]:
        """ 
        Returns the precision and recall of the model based on the predictions it has made
        
        Args:
            y_pred (ndarray) - a (121, m) array with the predictions of the model
            y_label (ndarray) - a (121, m) array with the actual values
            
        Returns:
            prediction_recall (list) - list of prediction recall values
        """
        m = y_label.shape[1]
        
        predictions = [interpret_probabilities(y_pred[:, i]) for i in range(m)]
        actual = [interpret_probabilities(y_label[:, i]) for i in range(m)]
        all_items = predictions.copy()
        all_items.extend(actual)
        labels = set(all_items)

        precisions = []
        
        for label in labels:
            total = 0
            false_positive = 0
            false_negative = 0
            true_positive = 0
            
            for i in range(m):
                entry = actual[i]
                pred = predictions[i]
                
                if entry == label:
                    total += 1
                    
                if entry == label and pred == entry:
                    true_positive += 1
                elif entry != label and pred == label:
                    false_positive += 1
            false_negative = total - true_positive
            
            precisions.append(PredictionRecall(
                t_p=true_positive,
                f_p=false_positive,
                f_n=false_negative,
                total=total,
                label=label
            ))
        
        return precisions
        
    def precision_recall(y_pred, y_label):
        """ 
            Print precision recall for model predictions
            
            Args:
                y_pred (ndarray): a (m, 121, 1) array with the predictions of the model
                y_label (ndarray): a (m, 121, 1) array with labels of the model
        """ 
        precisions = Evaluation.get_precision_recall(y_pred=y_pred, y_label=y_label)
        
        # Order precisions by totals
        n = len(precisions)
        for i in range(n):
            swapped = False
            
            for j in range(0, n-i-1):
                # Traverse array from 0 to n - i - 1
                # Swap if the element found is lesser than next element
                if precisions[j].total < precisions[j+1].total:
                    precisions[j], precisions[j + 1] = precisions[j + 1], precisions[j]
                    swapped = True
            
            if (swapped == False):
                break
        
        
        print(f"    Precision and recall")
        for prec in precisions:
            precision = 0 if prec.f_p + prec.t_p == 0 else (prec.t_p / (prec.f_p + prec.t_p))
            recall = 0 if prec.t_p + prec.f_n == 0 else (prec.t_p) / (prec.t_p + prec.f_n)
            p_1 = 0 if precision == 0 else 1 / precision
            r_1 = 0 if recall == 0 else 1 / recall
            f1_score = 1 / 0.5 * (p_1 + r_1)
            n_predictions = prec.t_p + prec.f_p
            print(f"{prec.label.center(5)}  TP:{prec.t_p}  FP:{prec.f_p}  FN:{prec.f_n}: Prec:{precision:.4f}  Rec:{recall:.4f}  F1:{f1_score:.4f}  #Pred: {n_predictions}  Sample:{prec.total}")
            
    def win_prediction_accuracy(y_pred, y_label):
        """ 
        Returns the ratio of accuratey predicted win / loss outcomes
        
        Args:
            y_pred (ndarray) - a (121, m) with model's predictions
            y_label (ndarray) - a (121, m) with actual labels
            
        Returns
            win_loss_accuracy (scalar) - ratio of correctly predicted wins / losses
        """
        win = 0
        draw = 1
        loss = 2
        
        predictions = np.argmax(y_pred, axis=0)
        predicted_wins = []
        actual = np.argmax(y_label, axis=0)
        actual_wins = []
        m = len(actual)
        
        for i in range(m):
            predicted_outcome = predictions[i]
            predicted_home = math.floor(predicted_outcome / 11)
            predicted_away = predicted_outcome % 11
            predicted_label = win if predicted_home > predicted_away else draw if predicted_home == predicted_away else loss
            predicted_wins.append(predicted_label)
            
            actual_outcome = actual[i]
            actual_home = math.floor(actual_outcome / 11)
            actual_away = actual_outcome % 11
            actual_label = win if actual_home > actual_away else draw if actual_home == actual_away else loss
            actual_wins.append(actual_label)
        
        return sum([predicted_wins[i] == actual_wins[i] for i in range(m)]) / m