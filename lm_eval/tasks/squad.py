import datasets
from lm_eval.base import rf, f1_score, mean
from . common import HFTask

class SQuAD(HFTask):
    DATASET_PATH = "squad_v2"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.data["train"]

    def validation_docs(self):
        return self.data["validation"]

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc):
        return 'Title: ' + doc['title'] + '\n\n' + 'Background: ' + doc['context'] + '\n\n' + 'Q: ' + doc['question'] + '\n\n' + 'A:'

    def doc_to_target(self, doc):
        answer_list = doc['answers']['text']
        if len(answer_list) > 0:
            answer = answer_list[0]
        else:
            answer = 'unanswerable'
        return " " + answer

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of 
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural 
            language description, as well as the few shot examples, and the question
            part of the document for `doc`. 
        """
        continuation = rf.greedy_until(ctx, ['\n'])
        return continuation
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        squad_metric = datasets.load_metric("squad_v2")

        predictions = {
            'id': doc['id'],
            'prediction_text': results[0],
        }

        references = {
            'id': doc['id'],
            'answers': doc['answers'],
        }

        metrics = squad_metric.compute(predictions=predictions, references=references)

        metrics.pop('total', None)
        metrics.pop('HasAns_total', None)
        metrics.pop('NoAns_total', None)
        metrics.pop('best_exact_thresh', None)
        metrics.pop('best_f1_thresh', None)

        return metrics

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        return { 
            'exact': mean, # Exact match (the normalized answer exactly match the gold answer)
            'f1': mean, #  The F-score of predicted tokens versus the gold answer
            'HasAns_exact': mean, # Exact match (the normalized answer exactly match the gold answer)
            'HasAns_f1': mean, # The F-score of predicted tokens versus the gold answer
            'NoAns_exact': mean, # Exact match (the normalized answer exactly match the gold answer)
            'NoAns_f1': mean, # The F-score of predicted tokens versus the gold answer
            'best_exact': mean, # Best exact match (with varying threshold)
            'best_f1': mean, # Best F1 (with varying threshold)
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        return { 
            'exact': True, # Exact match (the normalized answer exactly match the gold answer)
            'f1': True, #  The F-score of predicted tokens versus the gold answer
            'HasAns_exact': True, # Exact match (the normalized answer exactly match the gold answer)
            'HasAns_f1': True, # The F-score of predicted tokens versus the gold answer
            'NoAns_exact': True, # Exact match (the normalized answer exactly match the gold answer)
            'NoAns_f1': True, # The F-score of predicted tokens versus the gold answer
            'best_exact': True, # Best exact match (with varying threshold)
            'best_f1': True, # Best F1 (with varying threshold)
        }
