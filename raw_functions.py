import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from PyPDF2 import PdfReader
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
#from langdetect import detect
#import jieba
#import jieba.analyse
import nltk
import json

@st.cache_data
def download_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

def pdf_parser(input_pdf):
    pdf = PdfReader(input_pdf)
    pdf_content = ""
    for page in pdf.pages:
        pdf_content += page.extract_text()
    return pdf_content

# def langdetector

def get_keywords(file_paths): #这里的可以加一个条件判断，输入语言，判断分词器
    """
    这里的重点是，对每一个file做尽可能简短且覆盖全面的summarization
    """
    download_nltk()
    keywords_list = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = file.read()
            # tokenize
            words = word_tokenize(data)
            # remove punctuation
            words = [word for word in words if word.isalnum()]
            # remove stopwords
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]
            # lemmatization
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]
            # count word frequencies
            word_freq = Counter(words)
            # get top 20 most common words
            keywords = word_freq.most_common(20)
            new_keywords = []
            for word in keywords:
                new_keywords.append(word[0])
            str_keywords = ''
            for word in new_keywords:
                str_keywords += word + ", "
            keywords_list.append(f"Top20 frequency keywords for {file_path}: {str_keywords}")

    return keywords_list

def get_json_completion_from_messages(client, messages, model, temperature=0):
    client = client
    completion = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content

def get_visualize_stream_completion_from_messages(client, messages, model, temperature=0):
    message_placeholder = st.empty()
    client = client
    full_response = ""
    for response in client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    ):
        try:
            full_response += response.choices[0].delta.content
        except:
            full_response += ""
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    return full_response

def genarating_outline(client, keywords, num_lessons, language, model):
    system_message = 'You are a great AI teacher and linguist, skilled at create course outline based on summarized knowledge materials.'
    example_json = """{'outline':[['name_lesson1', 'abstract_lesson1'],['name_lesson2', 'abstract_lesson2']]}"""
    user_message = f"""You are a great AI teacher and linguist,
            skilled at generating course outline based on keywords of the course.
            Based on keywords provided, you should carefully design a course outline. 
            Requirements: Through learning this course, learner should understand those key concepts.
            Key concepts: {keywords}
            you should output course outline as a JSON object, Do not include anything else except that JSON object in your output.
            Example output format:{example_json}
            In the example, you can see each element in this JSON consists of two parts: the "name_lesson" part is the name of the lesson, and the "abstract_lesson" part is the one-sentence description of the lesson, intruduces knowledge it contained. 
            for each lesson in this course, you should provide these two information and organize them as exemplified. 
            for this course, you should design {num_lessons} lessons in total.
            the course outline should be written in {language}.
            Start the work now.
            """
    messages =  [
                {'role':'system',
                'content': system_message},
                {'role':'user',
                'content': user_message},
            ]

    response = get_json_completion_from_messages(client, messages, model)

    list_response = [['nothing in the answers..','please try again..']]
    try:
        list_response = json.loads(response)['outline']
    except SyntaxError:
        st.markdown('🤯Oops.. We encountered an error generating the outline of your course. Please try again.')
        pass
    return list_response

def chunkstring(string, length):
        return list((string[0+i:length+i] for i in range(0, len(string), length)))

def constructVDB(file_paths, collection_name='user_upload', embedding_function=SentenceTransformerEmbeddingFunction(model_name="paraphrase-mpnet-base-v2")):
    texts = ""
    for filename in file_paths:
        with open(filename, 'r') as f:
            content = f.read()
            texts += content
    chunks = chunkstring(texts, 1000) #1000 characters per chunk

    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)

    ids = [str(i) for i in range(len(chunks))]

    chroma_collection.add(ids=ids, documents=chunks)

    return chroma_collection

def searchVDB(query, chroma_collection):
    try:
        results = chroma_collection.query(query_texts=query, n_results=5, include=['documents', 'embeddings'])
        retrieved_chunks_list = results['documents'][0]

    except Exception:
        retrieved_chunks_list = []
        
    return retrieved_chunks_list

def write_one_lesson(client, topic, materials, language, style_options, ts_suggestion, model):
    system_message = 'You are a great AI teacher and linguist, skilled at writing informative and easy-to-understand course script based on given lesson topic and knowledge materials.'

    user_message = f"""You are a great AI teacher and linguist,
            skilled at writing informative and easy-to-understand course script based on given lesson topic and knowledge materials.\n
            You should write a course for new hands, they need detailed and vivid explaination to understand the topic. \n
            A high-quality course should meet requirements below:\n
            (1) Contains enough facts, data and figures to be convincing\n
            (2) The internal narrative is layered and logical, not a simple pile of items\n
            Make sure all these requirements are considered when writing the lesson script content.\n
            Please follow this procedure step-by-step when disgning the course:\n
            Step 1. Write down the teaching purpose of the lesson initially in the script. \n
            Step 2. Write content of the script. \n
            Your lesson topic and abstract is within the 「」 quotes, and the knowledge materials are within the 【】 brackets. \n
            lesson topic and abstract: 「{topic}」, \n
            knowledge materials related to this lesson：【{materials} 】 \n
            the script should be witten in {language}, and mathematical symbols should be written in markdown form. \n
            {style_options} \n
            {ts_suggestion} \n
            Start writting the script of this lesson now.
            """
    
    messages =  [
                {'role':'system',
                'content': system_message},
                {'role':'user',
                'content': user_message},
            ]

    response = get_visualize_stream_completion_from_messages(client, messages, model)
    return response

def decorate_user_question(user_question, retrieved_chunks_for_user):
    decorated_prompt = f'''You're a brilliant teaching assistant, skilled at answer stundent's question based on given materials.
    student's question: 「{user_question}」
    related materials:【{retrieved_chunks_for_user}】
    if the given materials are irrelavant to student's question, please use your own knowledge to answer the question.
    You need to break down the student's question first, find out what he really wants to ask, and then try your best to give a comprehensive answer.
    The language you're answering in should aligned with what student is using.
    Now you're talking to the student. Please answer.
    '''
    return decorated_prompt


def add_prompt_course_style(selected_style_list):
    initiate_prompt = 'Please be siginificantly aware that this course is requested to: \n'
    customize_prompt = ''
    if len(selected_style_list) != 0:
        customize_prompt += initiate_prompt
        for style in selected_style_list:
            if style == "More examples":
                customize_prompt += '- **contain more examples**. You should use your own knowledge to vividly exemplify key concepts occured in this course.\n'
            elif style == "More excercises":
                customize_prompt += '- **contain more excercises**. So last part of this lesson should be excercises.\n'
            elif style == "Easier to learn":
                customize_prompt += '- **Be easier to learn**. So you should use plain language to write the lesson script, and apply some metaphors & analogys wherever appropriate.\n'
    return customize_prompt

def teaching_supervision(outline, student_questions, client, model):
    system_message = 'You are a great AI teacher and teaching supervisor, skilled at giving teacher useful advice based on your insights towards students.'
    example_output = """{
                            'student_level': 'beginner', 
                            'student_interested': 'basic concepts',
                            'suggest_question': [
                                'Question_1',
                                'Question_2',
                                'Question_3'
                            ],
                            'script_revise_suggestions': 'please include more concrete examples in the lesson script.'
                        }"""
    user_message = f"""
                There's a student currently learning a course produced by your collegue and chatting with a teaching assistant.
                The teaching outline of the course is in the  {outline}. 
                During learning process, here are some questions this student asked the teaching assistant:
                student questions: 「{student_questions}」

                Please use your professional knowledge to recognize the learning status of the student and give support to student and teacher. 
                Please output your suggestions as a JSON object. Example output:
                {example_output}

                The four main keys in this JSON object are:
                1. "student_level": Determine the student's level of understanding of the course. Including three stages: "beginner", "intermediate" and "advanced".
                2. "student_interested": Determine which parts of the course student is more interested in. Including "basic concepts", "specific information in the material", "the whole idea of the material" and "professional topics that interest the student". Generally, beginners are interested in basic concepts and concrete info, while internediate and advanced learners are more likely to focus on the whole idea of the material or the professional topics suites their interests.
                3. "suggest_question": Recommend 3 questions based on student's preferences. If student is interested in basic concepts, recommend more explanations of basic concepts; If student is interested in their own professional topics, please recommend the extension of professional topics. Please install the three recommended questions in the json array. ⚠️ DO NOT repeat questions that students have already asked.
                4. "script_revise_suggestions": Please evaluate the amount of information in the lesson script, and how well it matches the preferences of the students, then give suggestions for revising the lesson script to make it more suited to student's preference.

                DO NOT include anything other than this JSON object in your output, and strictly follow the format of the output, especially the key names.
                """
    messages =  [
                    {'role':'system',
                    'content': system_message},
                    {'role':'user',
                    'content': user_message},
                ]
    
    response = get_json_completion_from_messages(client, messages, model)
    dict_response = json.loads(response)
    return dict_response

def add_prompt_ts_suggestions(student_level, student_interested, script_revise_suggestions):
    teaching_supervisor_suggestion = f""" 
        The current student you're teaching is in {student_level} level, and he/she is interested in {student_interested}. 
        Therefore I suggest "{script_revise_suggestions}".
    """
    return teaching_supervisor_suggestion

def decorate_suggested_questions(language, question_list):
    if language == 'Chinese':
        decorated_suggest_question = f"""
        根据您的学习状况，以下是为您推荐的问题:

        1. {question_list[0]}
        2. {question_list[1]}
        3. {question_list[2]}

        如果您需要，只需将它们复制到输入框中并询问我就好啦😊

        😉对了，根据对您学习状况的观察，OmniTutor调整了课堂脚本的写作方式。下一节课将更适合你学习。快试试吧~
        """
    elif language == 'English':
        decorated_suggest_question = f"""
        The following are recommended questions for you according to your learning status:

        1. {question_list[0]}
        2. {question_list[1]}
        3. {question_list[2]}

        If needed, simply copying them to your input box and ask me 😊

        😉 By the way, OmniTutor observed your learning and adjusted the way to write lesson script. The next lesson will be more suitable for you to study.
        """
    return decorated_suggest_question