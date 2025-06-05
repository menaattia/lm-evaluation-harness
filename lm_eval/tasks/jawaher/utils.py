# def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # def _process_doc(doc):
    #     ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    #     out_doc = {
    #         "query": preprocess(doc["activity_label"] + ": " + ctx),
    #         "choices": [preprocess(ending) for ending in doc["endings"]],
    #         "gold": int(doc["label"]),
    #     }
    #     return out_doc

    # return dataset.map(_process_doc)


# defines the input string a model will be given
# def doc_to_text(doc):
#   return  "Proverb: " + doc["Proverbs"]

# # can be either a text string that refers to the target string or an integer that 
# # refers to the index of the correct label
# def doc_to_target(doc):
#   return 0

# # When doc_to_target is set as an index, doc_to_choice must also be set with the
# # appropriate list of possible choice strings.
# def doc_to_choice(doc):
#   return [doc["Ar_Explanation"] or "", doc["Ar_Explanation" or ""]]


def doc_to_text(doc):
    """
    Create the input prompt for the model.
    """
    return f"Proverb: {doc['Proverbs']}\n\nWhat does this proverb mean?"

def doc_to_target(doc):
    """
    Return the index of the correct answer (0-based).
    Since we're setting the correct Arabic explanation as choice 0,
    the target is always 0.
    """
    return 0

def doc_to_choice(doc):
    """
    Return a list of multiple choice options.
    This creates choices by using the correct explanation and generating distractors.
    """
    correct_explanation = doc.get("Ar_Explanation", "")
    
    # Ensure we have a valid correct explanation
    if not correct_explanation or correct_explanation.strip() == "":
        correct_explanation = "No explanation available"
    
    # Return the correct answer as first choice, followed by incorrect
    choices = [correct_explanation.strip(), "هذا المثل يتحدث عن أهمية الصبر في الحياة"]
    
    return choices

def process_docs(dataset):
    """
    Optional: Process the entire dataset if needed.
    This is called once on the entire dataset before evaluation.
    """