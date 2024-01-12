import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
import tempfile
from PyPDF2 import PdfReader
import io
from sentence_transformers import SentenceTransformer
import time
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
#from langdetect import detect
#import jieba
#import jieba.analyse
import nltk

ss = st.session_state

@st.cache_data
def download_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

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

def get_completion_from_messages(client, messages, model, temperature=0):
    client = client
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content

def genarating_outline(client, keywords, num_lessons, language, model):
    system_message = 'You are a great AI teacher and linguist, skilled at create course outline based on summarized knowledge materials.'
    user_message = f"""You are a great AI teacher and linguist,
            skilled at generating course outline based on keywords of the course.
            Based on keywords provided, you should carefully design a course outline. 
            Requirements: Through learning this course, learner should understand those key concepts.
            Key concepts: {keywords}
            you should output course outline in a python list format, Do not include anything else except that python list in your output.
            Example output format:
            [[name_lesson1, abstract_lesson1],[name_lesson2, abstrct_lesson2]]
            In the example, you can see each element in this list consists of two parts: the "name_lesson" part is the name of the lesson, and the "abstract_lesson" part is the one-sentence description of the lesson, intruduces knowledge it contained. 
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

    response = get_completion_from_messages(client, messages, model)

    list_response = ['nothing in the answers..']
    try:
        list_response = eval(response)
    except SyntaxError:
        st.write('🤯Oops.. We encountered an error generating the outline of your course. Please try again.')
        pass
    return list_response

def constructVDB(file_paths):
    #把KM拆解为chunks
    chunks = []
    for filename in file_paths:
        with open(filename, 'r') as f:
            content = f.read()
            for chunk in chunkstring(content, 730):
                chunks.append(chunk)
    chunk_df = pd.DataFrame(chunks, columns=['chunk'])

    #从文本chunks到embeddings
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    embeddings = model.encode(chunk_df['chunk'].tolist())
    # convert embeddings to a dataframe
    embedding_df = pd.DataFrame(embeddings.tolist())
    # Concatenate the original dataframe with the embeddings
    paraphrase_embeddings_df = pd.concat([chunk_df, embedding_df], axis=1)
    # Save the results to a new csv file

    #从embeddings到向量数据库
    # Load the embeddings
    embeddings = paraphrase_embeddings_df.iloc[:, 1:].values  # All columns except the first (chunk text)
    # Ensure that the array is C-contiguous
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    # Preparation for Faiss
    dimension = embeddings.shape[1]  # the dimension of the vector space
    index = faiss.IndexFlatL2(dimension)
    # Normalize the vectors
    faiss.normalize_L2(embeddings)
    # Build the index
    index.add(embeddings)
    # write index to disk
    return paraphrase_embeddings_df, index

def searchVDB(search_sentence, paraphrase_embeddings_df, index):
    #从向量数据库中检索相应文段
    try:
        data = paraphrase_embeddings_df
        embeddings = data.iloc[:, 1:].values  # All columns except the first (chunk text)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        model = SentenceTransformer('paraphrase-mpnet-base-v2')
        sentence_embedding = model.encode([search_sentence])

        # Ensuring the sentence embedding is in the correct format
        sentence_embedding = np.ascontiguousarray(sentence_embedding, dtype=np.float32)
        # Searching for the top 3 nearest neighbors in the FAISS index
        D, I = index.search(sentence_embedding, k=3)
        # Printing the top 3 most similar text chunks
        retrieved_chunks_list = []
        for idx in I[0]:
            retrieved_chunks_list.append(data.iloc[idx].chunk)

    except Exception:
        retrieved_chunks_list = []
        
    return retrieved_chunks_list

def write_one_lesson(client, topic, materials, language, style_options):
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
            Start writting the script of this lesson now.
            """
    
    messages =  [
                {'role':'system',
                'content': system_message},
                {'role':'user',
                'content': user_message},
            ]

    response = get_completion_from_messages(client, messages)
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

def initialize_file(added_files):
    temp_file_paths = []
    with st.spinner('Processing file...'):
        for added_file in added_files:
            if added_file.name.endswith(".pdf"):
                string = pdf_parser(added_file)
                with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
                    tmp.write(string.encode("utf-8"))
                    tmp_path = tmp.name
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
                    tmp.write(added_file.getvalue())
                    tmp_path = tmp.name
            temp_file_paths.append(tmp_path)
    success_outline = st.empty()
    success_outline = st.success('Processing file...Done')
    time.sleep(0.5)
    success_outline.empty()
    return temp_file_paths

def initialize_vdb(temp_file_paths):
    with st.spinner('Constructing vector database from provided materials...'):
        embeddings_df, faiss_index = constructVDB(temp_file_paths)
    st.success("Constructing vector database from provided materials...Done")
    return embeddings_df, faiss_index

def initialize_outline(client, temp_file_paths, num_lessons, language, model):
    with st.spinner('Generating Course Outline...'):
        summarized_materials = get_keywords(temp_file_paths)
        course_outline_list = genarating_outline(client, summarized_materials, num_lessons, language, model)
    st.success("Generating Course Outline...Done")
    course_outline_string = ''
    lessons_count = 0
    for outline in course_outline_list:
        lessons_count += 1
        course_outline_string += f"{lessons_count}." + outline[0]
        course_outline_string += '\n\n' + outline[1] + '\n\n'
    with st.expander("Check the course outline", expanded=False):
        st.write(course_outline_string)
    
    return course_outline_list

## Initialize content

def regenerate_outline(course_outline_list):
    try:
        course_outline_string = ''
        lessons_count = 0
        for outline in course_outline_list:
            lessons_count += 1
            course_outline_string += f"{lessons_count}." + outline[0]
            course_outline_string += '\n\n' + outline[1] + '\n\n'
        with st.expander("Check the course outline", expanded=False):
            st.write(course_outline_string)
    except Exception:
        pass

## Regenerate content

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

def initialize_session_state():
    """
    All variables needed to be stored across reruns
    """
    if "embeddings_df" not in ss:
        ss.embeddings_df = ''
    if "faiss_index" not in ss:
        ss.faiss_index = ''
    if "course_outline_list" not in ss:
        ss.course_outline_list = []
    if "course_content_list" not in ss:
        ss.course_content_list = []
    if "OPENAI_API_KEY" not in ss:
        ss["OPENAI_API_KEY"] = ''
    if "openai_model" not in ss:
        ss["openai_model"] = 'gpt-4-1106-preview' 
    if "messages_ui" not in ss:
        ss.messages_ui = []
    if "messages" not in ss:
        ss.messages = []

    if "language" not in ss:
        ss.language = ''
    if "style_options" not in ss:
        ss.style_options = ''

def app():
    st.title("OmniTutor 2.0")
    st.header("Your personalized AI Knowledge Engine 🦉")
    st.markdown("""
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                right: 10px;
                width: auto;
                background-color: transparent;
                text-align: right;
                padding-right: 10px;
                padding-bottom: 10px;
            }
        </style>
        <div class="footer">Made with 🧡 by Siyuan</div>
    """, unsafe_allow_html=True)
    with st.sidebar:
        api_key = st.text_input('🔑 Your OpenAI API key:', 'sk-...')
        use_35 = st.toggle('Use GPT-3.5 (GPT-4 is default)')
        if use_35:
            ss["openai_model"] = 'gpt-3.5-turbo-1106'
        st.image("https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/WechatIMG1729.jpg")
        added_files = st.file_uploader('⏫ Upload .md or .pdf files, simultaneous mixed upload both types is supported.', type=['.md','.pdf'], accept_multiple_files=True)
        with st.expander('⚙️ Customize my course'):
            num_lessons = st.slider('How many lessons do you want this course to have?', min_value=2, max_value=15, value=5, step=1)
            custom_options = st.multiselect(
                'Preferred teaching style :grey[(Recommend new users not to select)]',
                ['More examples', 'More excercises', 'Easier to learn'],
                max_selections = 2
            )
            ss.style_options = add_prompt_course_style(custom_options)
            ss.language = 'English'
            Chinese = st.checkbox('Output in Chinese')
            if Chinese:
                ss.language = 'Chinese'
        btn = st.button('Start learning!')

    initialize_session_state()

    ss.description = st.markdown('''
        <font color = 'grey'> An all-round teacher. A teaching assistant who really knows the subject. **Anything. Anywhere. All at once.** </font> :100:
        
        Github Repo: https://github.com/Siyuan-Harry/OmniTutor 

        ### ✨ Key features                                           
                                                   
        - 🧑‍🏫 **Concise and clear course creation**: <font color = 'grey'>Generated from your learning notes (**.md**) or any learning materials (**.pdf**)!</font>
        - 📚 **All disciplines**: <font color = 'grey'>Whether it's math, physics, literature, history or coding, OmniTutor covers it all.</font>
        - ⚙️ **Customize your own course**: <font color = 'grey'>Choose your preferred teaching style, lesson count and language.</font>
        - ⚡️ **Fast respond with trustable accuracy**: <font color = 'grey'>Problem-solving chat with the AI teaching assistant who really understand the materials.</font>
        
        ### 🏃‍♂️ Get started!
                                                    
        1. **Input Your OpenAI API Key**: <font color = 'grey'>Give OmniTutor your own OpenAI API key (On top of the **sidebar**) to get started.</font>
        2. **Upload learning materials**: <font color = 'grey'>The upload widget in the sidebar supports PDF and .md files simutaenously.</font>
        3. **Customize your course**: <font color = 'grey'>By few clicks and swipes, adjusting teaching style, lesson count and language for your course.</font>
        4. **Start course generating**: <font color = 'grey'>Touch "Generate my course!" button in the sidebar, then watch how OmniTutor creates personal-customized course for you.</font>
        5. **Interactive learning**: <font color = 'grey'>Learn the course, and ask OmniTutor any questions related to this course whenever you encountered them.</font>
                                
        🎉 Have fun playing with Omnitutor!
                                                                                                                                       
        ''', unsafe_allow_html=True
        )
    
    user_question = st.chat_input("Enter your questions when learning...")
    if added_files:
        temp_file_paths = initialize_file(added_files)
        st.session_state.embeddings_df, st.session_state.faiss_index = initialize_vdb(temp_file_paths)

    if btn:
        if api_key != "sk-..." and api_key !="" and api_key.startswith("sk-"):
            ss.description.empty()
            ss["OPENAI_API_KEY"] = api_key
            client = OpenAI(api_key = ss["OPENAI_API_KEY"])
            col1, col2 = st.columns([0.6,0.4])
            ss.course_outline_list = initialize_outline(client, temp_file_paths, num_lessons, ss.language)

        else:
            ss.description.empty()
            st.write("请输入正确的OpenAI API Key令牌")
    
    

        

    