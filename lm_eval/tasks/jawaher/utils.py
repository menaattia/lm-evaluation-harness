
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
        prompt = (
        "You are tasked with selecting the correct explanation for the following proverb.\n"
        "Choose the correct explanation from the options provided. Only output the letter corresponding to the correct answer and nothing else.\n\n"
        f"Proverb: {doc['Proverb']}\n\n"
        f"Options: A. {correct_explanation} \nB. {incorrect_explanation}\n"
        "Answer: "
        )  
    elif order == "B":
        prompt = (
        "You are tasked with selecting the correct explanation for the following proverb.\n"
        "Choose the correct explanation from the options provided. Only output the letter corresponding to the correct answer and nothing else.\n\n"
        f"Proverb: {doc['Proverb']}\n\n"
        f"Options: A. {incorrect_explanation} \nB. {correct_explanation}\n"
        "Answer: "
        )     
    else:
        prompt = (
        "You are tasked with selecting the correct explanation for the following proverb.\n"
        "Choose the correct explanation from the options provided. Only output the letter corresponding to the correct answer and nothing else.\n\n"
        f"Proverb: {doc['Proverb']}\n\n"
        f"Options: A. {doc['Options'][0]} \nB. {doc['Options'][1]}\n"
        "Answer: "
        )
    return prompt

def doc_to_text_incorrect(doc):
    return (
        "You are tasked with selecting the incorrect explanation for the following proverb.\n"
        "Choose the incorrect explanation from the options provided. Only output the letter corresponding to the incorrect answer and nothing else.\n\n"
        f"Proverb: {doc['Proverb']}\n\n"
        f"Options: A. {doc['Options'][0]} \nB. {doc['Options'][1]}\n"
        "Answer: "
        )

def doc_to_text_incorrect_idiom(doc):
    return (
        "You are tasked with selecting the incorrect explanation for the following idiom.\n"
        "Choose the incorrect explanation from the options provided. Only output the letter corresponding to the incorrect answer and nothing else.\n\n"
        f"Idiom: {doc['Idiom']}\n\n"
        f"Options: A. {doc['Options'][0]} \nB. {doc['Options'][1]}\n"
        "Answer: "
        )

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

def doc_to_text_prag_use(doc):
    prompt = (
        "Your task is to fill in the blank with the correct idiom.\n"
        "Choose the correct idiom from the options provided. Only output the letter corresponding to the correct answer and nothing else.\n\n"
        f"Sentence: {doc['sentence']}\n\n"
        f"Options: A. {doc['Options'][0]} \nB. {doc['Options'][1]}\n"
        "Answer: "
        )

    return prompt

def doc_to_text_prag_use_proverb(doc):
    prompt = (
        "Your task is to fill in the blank with the correct proverb.\n"
        "Choose the correct proverb from the options provided. Only output the letter corresponding to the correct answer and nothing else.\n\n"
        f"Conversation: {doc['conversation']}\n\n"
        f"Options: A. {doc['Options'][0]} \nB. {doc['Options'][1]}\n"
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

def doc_to_target_incorrect(doc):
    """
    Return the incorrect answer. Only 2 choices.
    """
    if doc['Answer']== 0:
        return "B"
    else:
        return "A"
    

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

def process_docs_prag_use(dataset):
    """
    This is called once on the entire dataset before evaluation.
    """

    def _shuffle_explanations(doc):
        correct = doc["correct"]
        incorrect = doc["incorrect"]

        options = [correct, incorrect]
        random.shuffle(options)

        correct_index = options.index(correct)

        return {
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

def create_maps_completion_dataset(dataset):
    """
    Convert MAPS dataset to proverb completion format
    """
    def _incomplete(doc):
        proverb = doc['proverb']
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
    
    prompt = (
        "You are tasked with selecting the correct explanation for the following idiom.\n"
        "Choose the correct explanation from the options provided. Only output the letter corresponding to the correct answer and nothing else.\n\n"
        f"Idiom: {doc['Idiom']}\n\n"
        f"Options: A. {doc['Options'][0]} \nB. {doc['Options'][1]}\n"
        "Answer: "
        )
    return prompt

def doc_to_text_idioms_context(doc):
    """
    Create the input prompt for the model.
    """  
    prompt = (
        "You are tasked with selecting the correct explanation for the following idiom, given the idiom in a sentence for context.\n"
        "Choose the correct explanation from the options provided. Only output the letter corresponding to the correct answer and nothing else.\n\n"
        f"Idiom: {doc['Idiom']}\n\n"
        f"Sentence: {doc['full_sentence']}\n\n"
        f"Options: A. {doc['Options'][0]} \nB. {doc['Options'][1]}\n"
        "Answer: "
        )
    return prompt

def process_docs_idioms(dataset):
    """
    This is called once on the entire dataset before evaluation.
    """

    def _shuffle_explanations(doc):
        correct = doc["Ar_Explanation"]
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

def doc_to_text_complete(doc):
    # " أكمل المثل التالي بالكلمة الصحيحة فقط. اكتب كلمة واحدة: \n{{ incomplete_proverb }}"
    prompt = (
        "You are tasked with completing the proverb with the last word. Output the next word only. \n\n"
        f"Incomplete Proverb: {doc['incomplete_proverb']}\n"
        "Answer: "
    )
    return prompt


def doc_to_sentiment_prompts(doc):
    prompt1 = (
        f"Determine the sentiment of the following Arabic proverb as either Positive, Negative, or Neutral.\n\n"
        f"Proverb: {doc['Proverbs']}\n"
        f"Sentiment:"
    )
    prompt2 = (
        f"Determine the sentiment of the following explanation as either Positive, Negative, or Neutral.\n\n"
        f"Explanation: {doc['Ar_Explanation']}\n"
        f"Sentiment:"
    )
    return [prompt1, prompt2]

# utils.py

# def create_sentiment_match_docs(dataset):
#     """
#     For each sample, yield two docs:
#     - one for proverb sentiment
#     - one for explanation sentiment
#     Each doc has a 'phase' key: 'proverb' or 'explanation'
#     Grouping is by index.
#     """
#     for i, sample in enumerate(dataset):
#         yield {
#             'id': i,
#             'phase': 'proverb',
#             'text': sample['proverb'],
#             'explanation': sample['explanation'],
#         }
#         yield {
#             'id': i,
#             'phase': 'explanation',
#             'text': sample['explanation'],
#             'proverb': sample['proverb'],
#         }


def doc_to_text_proverb_sentiment(doc):
    # return (
    #     f"Determine the sentiment of the following Arabic proverb as either Positive, Negative, or Neutral.\n\n"
    #     "Only output the sentiment and nothing else.\n"
    #     f"Proverb: {doc['Proverbs']}\n"
    #     f"Sentiment:"
    # )
    return (
        "Determine the connotation of the following Arabic proverb. Classify the connotation as Positive, Negative, or Neutral based on the following guidelines:\n\n"
        "Positive Connotation: It conveys optimism, hope, praise, or beneficial outcomes. It highlights virtues such as kindness, success, loyalty, or happiness. It encourages or celebrates desirable behaviors or outcomes.\n"
        "Negative Connotation: It expresses pessimism, caution, loss, or undesirable consequences. It highlights flaws, mistakes, or risks and often reflects on the dangers or negative results of certain actions.\n"
        "Neutral Connotation: It provides general advice or observation without invoking strong feelings or judgment.\n"
        f"Proverb: {doc['Proverbs']}\n"
        "Only output the connotation and nothing else.\n\n"
        "Connotation:\n"
    )

def doc_to_text_idiom_sentiment(doc):
    return (
        "Determine the connotation of the following Arabic idiom. Classify the connotation as Positive, Negative, or Neutral based on the following guidelines:\n\n"
        "Positive Connotation: It conveys optimism, hope, praise, or beneficial outcomes. It highlights virtues such as kindness, success, loyalty, or happiness. It encourages or celebrates desirable behaviors or outcomes.\n"
        "Negative Connotation: It expresses pessimism, caution, loss, or undesirable consequences. It highlights flaws, mistakes, or risks and often reflects on the dangers or negative results of certain actions.\n"
        "Neutral Connotation: It provides general advice or observation without invoking strong feelings or judgment.\n"
        f"Idiom: {doc['Idiom']}\n"
        "Only output the connotation and nothing else.\n\n"
        "Connotation:\n"
    )

def doc_to_text_expl_sentiment(doc):
    # return (
    #     f"Determine the sentiment of the following explanation as either Positive, Negative, or Neutral.\n\n"
    #     "Only output the sentiment and nothing else.\n"
    #     f"Explanation: {doc['Ar_Explanation']}\n"
    #     f"Sentiment:"
    # )
    return (
        "Determine the connotation of the following Arabic explanation. Classify the connotation as Positive, Negative, or Neutral based on the following guidelines:\n\n"
        "Positive Connotation: It conveys optimism, hope, praise, or beneficial outcomes. It highlights virtues such as kindness, success, loyalty, or happiness. It encourages or celebrates desirable behaviors or outcomes.\n"
        "Negative Connotation: It expresses pessimism, caution, loss, or undesirable consequences. It highlights flaws, mistakes, or risks and often reflects on the dangers or negative results of certain actions.\n"
        "Neutral Connotation: It provides general advice or observation without invoking strong feelings or judgment.\n"
        f"Explanation: {doc['Ar_Explanation']}\n"
        "Only output the connotation and nothing else.\n\n"
        "Connotation:\n"
    )

def doc_to_choice_sentiment_mcq(doc):
    return ["Positive", "Negative", "Neutral"]

def doc_to_target_sentiment_mcq(doc):
    # Make sure the capitalization matches your choices
    return doc["Sentiment"].capitalize()

def doc_to_text_gen(doc):
    prompt = (
        "Your task is to explain the meaning of the following Arabic proverb. Provide a clear and concise explanation in Arabic, highlighting its figurative meaning and any cultural or contextual significance.\n"
        "Only output the Arabic explanation and nothing else.\n\n"
        f"Proverb: {doc['Proverbs']}\n\n"
        "Arabic Explanation:\n"
    )
    return prompt

def doc_to_text_gen_idiom(doc):
    prompt = (
        "Your task is to explain the meaning of the following Arabic idiom. Provide a clear and concise explanation in Arabic, highlighting its figurative meaning and any cultural or contextual significance.\n"
        "Only output the Arabic explanation and nothing else.\n\n"
        f"Idiom: {doc['Idiom']}\n\n"
        "Arabic Explanation:\n"
    )
    return prompt