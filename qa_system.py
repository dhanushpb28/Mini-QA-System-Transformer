# Mini Question-Answering System using Pretrained Transformer (BERT)

from transformers import pipeline

# Load a more powerful pretrained model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Health domain context (long passage)
context = """
Mental health is an essential component of overall health and well-being. It affects how people think, feel, and act in daily life. 
Mental health determines how individuals handle stress, relate to others, and make choices. Common mental health disorders include 
depression, anxiety, bipolar disorder, and schizophrenia. Depression is one of the most common conditions, often characterized by 
persistent sadness, loss of interest, and low energy. Anxiety disorders, on the other hand, involve excessive fear or worry that can 
interfere with normal activities. While genetics and brain chemistry play a role in mental health disorders, external factors such as 
trauma, abuse, social isolation, and chronic stress can also contribute.

Treatment and management of mental health conditions involve a combination of therapies, medications, and lifestyle changes. 
Psychotherapy, also known as talk therapy, allows patients to discuss their thoughts and emotions with trained professionals. 
Cognitive Behavioral Therapy (CBT) is one of the most effective approaches used to change negative thinking patterns. Antidepressants, 
mood stabilizers, and anti-anxiety drugs are commonly prescribed medications. Apart from medical treatment, lifestyle improvements 
such as exercise, meditation, maintaining social connections, and having a balanced diet greatly help in managing symptoms.

The role of technology in mental health care has grown rapidly. AI-powered chatbots, online counseling platforms, and mental health 
tracking apps have made professional help more accessible and affordable. Wearable devices now monitor physiological signals like 
heart rate and sleep patterns to detect early signs of stress or depression. Schools and workplaces have started mental health 
awareness programs to reduce stigma and promote open discussions. Governments and organizations worldwide are working toward 
integrating mental health into primary care systems to ensure early intervention. Despite these efforts, mental health care remains 
underfunded in many regions, and millions of people still lack access to proper treatment. Achieving global mental well-being requires 
collaboration among healthcare providers, policymakers, educators, and communities.
"""

# Ask multiple questions
while True:
    question = input("\nAsk a question about mental health (or type 'exit' to quit): ")
    if question.lower() == "exit":
        break
    result = qa_pipeline(question=question, context=context)
    print(f"Answer: {result['answer']} (Confidence: {result['score']:.2f})")
