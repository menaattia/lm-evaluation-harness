
import random 

random.seed(0)

def doc_to_text(doc, order):
    """
    Create the input prompt for the model. The prompt contains 2 answer choices. 
    order: the order of the correct answer choice. "A" if the correct option is always first
    in the prompt, "B" if the correct option is always second, "random" if it is random
    """

    correct_explanation = doc.get("Ar_Explanation", "")
    incorrect_explanation = doc.get("Incorrect_Explanation", "")
    
    # Ensure we have a valid correct explanation
    if not correct_explanation or correct_explanation.strip() == "":
        correct_explanation = "No explanation available"
    
    if order == "A":
        prompt = f"You are tasked with selecting the correct explanation for the following proverb. Only output the letter corresponding to the correct answer. \nProverb: {doc['Proverbs']}\n\nChoose the correct explanation from the options provided. \n\nOptions: A. {correct_explanation} \nB. {incorrect_explanation}"  
    elif order == "B":
        prompt = f"You are tasked with selecting the correct explanation for the following proverb. Only output the letter corresponding to the correct answer. \nProverb: {doc['Proverbs']}\n\nChoose the correct explanation from the options provided. \n\nOptions: A. {incorrect_explanation} \nB. {correct_explanation}"     
    else:
        prompt = (
        "You are tasked with selecting the correct explanation for the following proverb.\n"
        "Choose the correct explanation from the options provided. Only output the letter corresponding to the correct answer.\n\n"
        f"Proverb: {doc['Proverb']}\n\n"
        f"Options: A. {doc['Options'][0]} \nB. {doc['Options'][1]}\n"
        "Answer: "
        )
    return prompt


def doc_to_text_a(doc):
    return doc_to_text(doc, "A")

def doc_to_text_b(doc):
    return doc_to_text(doc, "B")

def doc_to_text_random(doc):
    return doc_to_text(doc, "random")

def doc_to_text4(doc):
    """
    Create the input prompt for the model. The prompt contains 4 answer choices. 
    The answer choices are in random order. 
    """
    prompt = (
        "You are tasked with selecting the correct explanation for the following proverb.\n"
        "Choose the correct explanation from the options provided. Only output the letter corresponding to the correct answer.\n\n"
        f"Proverb: {doc['Proverb']}\n\n"
        f"Options: A. {doc['Options'][0]} \nB. {doc['Options'][1]}\nC. {doc['Options'][2]}\nD. {doc['Options'][3]}\n"
        "Answer: "
        )

    return prompt

def doc_to_target(doc):
    """
    Return the correct answer.
    """
    if doc['Answer']== 0:
        return "A"
    elif doc['Answer']== 1:
        return "B"
    elif doc['Answer']== 2:
        return "C"
    else:
        return "D"

def doc_to_choice(doc):
    """
    Return a list of multiple choice options for 4 options.
    """
    
    return ["A", "B", "C", "D"]


def doc_to_choice2(doc):
    """
    Return a list of multiple choice options for 2 options.
    """
    
    return ["A", "B"]


def process_docs(dataset):
    """
    This is called once on the entire dataset before evaluation.
    """

    def _shuffle_explanations(doc):
        correct = doc["Ar_Explanation"]
        incorrect = doc["Incorrect_Explanation"]
        # incorrect2 = doc["Incorrect_Explanation2"]
        # incorrect3 = doc["Incorrect_Explanation3"]
        # incorrect = doc["shuffled1"]
        # incorrect2 = doc["shuffled2"]
        # incorrect3 = doc["shuffled3"]

        options = [correct, incorrect]
        # options = [correct, incorrect, incorrect2, incorrect3]
        random.shuffle(options)

        correct_index = options.index(correct)

        return {
            "Proverb": doc["Proverbs"],
            "Options": options,
            "Answer": correct_index
        }

    return dataset.map(_shuffle_explanations)

def process_maps(dataset):
    """
    This is called once on the entire dataset before evaluation.
    """

    def _touppercase(doc):
        
        return {
            "answer_key": doc["answer_key"].upper(),
        }

    return dataset.map(_touppercase)

def create_proverb_completion_dataset(dataset):
    """
    Convert Jawaher dataset to proverb completion format
    """
    def _incomplete(doc):
        proverb = doc['Proverbs']
        words = proverb.split()
        
        incomplete_proverb = ""
        last_word = ""

        if len(words) >= 2:  # Ensure we have at least 2 words
            incomplete_proverb = ' '.join(words[:-1])
            last_word = words[-1]
            
        return{
            'proverb': proverb,
            'incomplete_proverb': incomplete_proverb,
            'last_word': last_word,
        }   
    
    return dataset.map(_incomplete)


def doc_to_text_idioms(doc):
    """
    Create the input prompt for the model.
    """

    correct_explanation = doc.get("Explanation", "")
    incorrect_explanation = doc.get("Incorrect_Explanation", "")
    
    # Ensure we have a valid correct explanation
    if not correct_explanation or correct_explanation.strip() == "":
        correct_explanation = "No explanation available"
    
    # if order == "A":
    #     prompt = f"You are tasked with selecting the correct explanation for the following idiom. \nProverb: {doc['Idiom']}\n\nChoose the correct explanation from the options provided. \n\nOptions: A. {correct_explanation} \nB. {incorrect_explanation}"  
    # elif order == "B":
    #     prompt = f"You are tasked with selecting the correct explanation for the following idiom. \nProverb: {doc['Idiom']}\n\nChoose the correct explanation from the options provided. \n\nOptions: A. {incorrect_explanation} \nB. {correct_explanation}"     
    # else:
    prompt = (
        "You are tasked with selecting the correct explanation for the following proverb.\n"
        "Choose the correct explanation from the options provided. Only output the letter corresponding to the correct answer.\n\n"
        f"Idiom: {doc['Idiom']}\n\n"
        f"Options: A. {doc['Options'][0]} \nB. {doc['Options'][1]}\n"
        "Answer: "
        )
    return prompt


def process_docs_idioms(dataset):
    """
    This is called once on the entire dataset before evaluation.
    """

    def _shuffle_explanations(doc):
        correct = doc["Explanation"]
        incorrect = doc["Incorrect_Explanation"]


        options = [correct, incorrect]
        random.shuffle(options)

        correct_index = options.index(correct)

        return {
            "Idiom": doc["Idiom"],
            "Options": options,
            "Answer": correct_index
        }

    return dataset.map(_shuffle_explanations)

def doc_to_text_maps(doc):
    prompt = (
        "You are tasked with selecting the correct explanation for the following proverb.\n"
        "Choose the correct explanation from the options provided. Only output the letter corresponding to the correct answer.\n\n"
        f"Proverb: {doc['proverb']}\n\n"
        f"Options: A. {doc['answer1']} \nB. {doc['answer2']}\n"
        "Answer: "
        )
    return prompt

def doc_to_text_maps_context(doc):
    prompt = (
        "You are tasked with selecting the correct explanation for the following proverb.\n"
        "Choose the correct explanation from the options provided. Only output the letter corresponding to the correct answer.\n\n"
        f"Proverb: {doc['proverb']}\n\n"
        f"Context: {doc['conversation']}\n\n"
        f"Options: A. {doc['answer1']} \nB. {doc['answer2']}\n"
        "Answer: "
        )
    return prompt