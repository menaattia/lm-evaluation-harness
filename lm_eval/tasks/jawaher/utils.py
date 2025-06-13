
import random 

random.seed(0)

def doc_to_text(doc):
    """
    Create the input prompt for the model.
    """

    # correct_explanation = doc.get("Ar_Explanation", "")
    # incorrect_explanation = doc.get("Incorrect_Explanation", "")
    
    # # Ensure we have a valid correct explanation
    # if not correct_explanation or correct_explanation.strip() == "":
    #     correct_explanation = "No explanation available"
    
    prompt = f"You are tasked with selecting the correct explanation for the following proverb. \nProverb: {doc['Proverb']}\n\nChoose the correct explanation from the options provided. \n\nOptions: A. {doc['Options'][0]} \nB. {doc['Options'][1]}"

    return prompt

def doc_to_target(doc):
    """
    Return the correct answer.
    """
    # return "B"
    return "A" if doc['Answer']== 0 else "B"

def doc_to_choice(doc):
    """
    Return a list of multiple choice options.
    """
    
    return ["A", "B"]


def process_docs(dataset):
    """
    Optional: Process the entire dataset if needed.
    This is called once on the entire dataset before evaluation.
    """

    def _shuffle_explanations(doc):
        correct = doc["Ar_Explanation"]
        incorrect = doc["Incorrect_Explanation"]

        options = [correct, incorrect]
        random.shuffle(options)

        correct_index = options.index(correct)

        return {
            "Proverb": doc["Proverbs"],
            "Options": options,
            "Answer": correct_index
        }

    return dataset.map(_shuffle_explanations)