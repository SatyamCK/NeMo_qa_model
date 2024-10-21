import nemo.collections.nlp as nemo_nlp
from transformers import BertTokenizer
import torch


# Load pretrained question-answering model from NeMo
qa_model = nemo_nlp.models.QAModel.from_pretrained(model_name="qa_squadv1_1_bertlarge")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# # Fine-tune the model on your dataset
# qa_model.train_model(training_data='path_to_training_data')


def answer_question(context, question):
    """
    Given a context and a question, this function returns an answer
    using a NeMo pretrained model.
    """

    prompt = f"""
        You are a question-answering assistant. Please read the following context carefully and answer the user's question based on the information provided. Make sure your response is clear and directly related to the context.

        Context: {context}

        User Question: {question}
    """
    # Tokenize the input
    inputs = tokenizer(
        text=prompt,
        text_pair=context,
        return_tensors="pt",
        max_length=512,  # Ensure inputs are within the model's token limit
        truncation=True,  # Truncate inputs that are too long
        padding="max_length"  # Pad inputs to the maximum length
    )
    # print(inputs)
    # Get the answer from the NeMo model
    outputs = qa_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs["token_type_ids"]
    )

    # Split the outputs into start_logits and end_logits
    start_scores, end_scores = outputs.split(1, dim=-1)

    # Remove the extra last dimension
    start_scores = start_scores.squeeze(-1)
    end_scores = end_scores.squeeze(-1)

    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores, dim=-1)
    answer_end = torch.argmax(end_scores, dim=-1) + 1

    # Ensure answer_end does not go out of bounds
    answer_end = min(answer_end.item(), inputs["input_ids"].shape[1] - 1)

    # Convert the input_ids back to tokens using the tokenizer
    answer_tokens = inputs["input_ids"][0][answer_start:answer_end]

    # Convert tokens back into words using the tokenizer
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer.strip() # Return the answer without extra spaces

if __name__ == "__main__":
    context = """
    In the 21st century, technology has rapidly transformed the way we live, work, and interact with one another.
    The rise of the internet and smartphones has revolutionized communication and created new opportunities.
    However, it also raises concerns about privacy, mental health, and the ethical implications of artificial intelligence.
    """

    print("Context: \n", context)
    # prompt = "sAsk a question about the context above to get a relevant answer."
    while True:
        question = input("\nEnter your question (type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        # Get the answer using the NeMo model
        answer = answer_question(context, question)
        print("Answer: ", answer)
