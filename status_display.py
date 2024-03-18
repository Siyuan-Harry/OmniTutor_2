from raw_functions import *
import tempfile
import time

ss = st.session_state

def initialize_session_state():
    """
    All variables needed to be stored across reruns
    """
    if "temp_file_paths" not in ss:
        ss.temp_file_paths = ''
    if "chroma_collection" not in ss:
        ss.chroma_collection = ''
    if "course_outline_list" not in ss:
        ss.course_outline_list = []
    if "course_content_list" not in ss:
        ss.course_content_list = []
    if "lesson_counter" not in ss:
        ss.lesson_counter = 0

    if "OPENAI_API_KEY" not in ss:
        ss["OPENAI_API_KEY"] = ''
    if "openai_model" not in ss:
        ss["openai_model"] = 'gpt-4-turbo-preview' 
    if "client" not in ss:
        ss.client = ''
    if "messages_ui" not in ss:
        ss.messages_ui = []
    if "messages" not in ss:
        ss.messages = []
    if "user_message_count" not in ss:
        ss.user_message_count = 0

    if "num_lessons" not in ss:
        ss.num_lessons = 0
    if "learning_intention" not in ss:
        ss.learning_intention = ''
    if "language" not in ss:
        ss.language = ''
    if "style_options" not in ss:
        ss.style_options = ''
    if "ts_suggestions" not in ss:
        ss.ts_suggestions = ''

    if "start_learning" not in ss:
        ss.start_learning = 0
    if "main_page_displayed" not in ss:
        ss.main_page_displayed = True
    if "chatInput_displayed" not in ss:
        ss.chatInput_displayed = False

def initialize_file(added_files):
    temp_file_paths = []
    with st.spinner('Processing file(s)...'):
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
    st.success('Processing file(s)...Done')
    #time.sleep(0.5)
    #success_outline.empty()
    return temp_file_paths

# 这里以及下面都要改. 前后端分离
def initialize_vdb(temp_file_paths):
    with st.spinner('Constructing vector database from provided materials...'):
        chroma_collection = constructVDB(temp_file_paths)
    st.success("Constructing vector database from provided materials...Done")
    return chroma_collection

def initialize_outline(client, temp_file_paths, learning_intention, num_lessons, language, model):
    with st.spinner('Generating Course Outline...'):
        summarized_materials = get_keywords(temp_file_paths)
        course_outline_list = genarating_outline(client, summarized_materials, learning_intention, num_lessons, language, model)
    st.success("Generating Course Outline...Done")
    course_outline_string = ''
    lessons_count = 0
    for outline in course_outline_list:
        lessons_count += 1
        course_outline_string += f"**{lessons_count}. {outline[0]}**"
        course_outline_string += f"\n\n{outline[1]} \n\n"
    with st.expander("Check the course outline", expanded=False):
        st.markdown(course_outline_string)
    return course_outline_list

def visualize_new_content(client, count_generating_content, lesson_description, chroma_collection, language, style_options, ts_suggestions, model):
    with st.expander(f"Learn the lesson {count_generating_content} ", expanded=False):
        with st.spinner(text = "Extending the course outline to enhance the course content retrieval..."):
            dict_augmentedQueries = augment_multiple_query(ss.client, lesson_description)
            
            list_augmentedQueries = dict_augmentedQueries['suggested_questions']     
            
            queries = [lesson_description] + list_augmentedQueries

        st.success('Extending the course outline to enhance the course content retrieval...Done!')

        retrievedChunksList = searchVDB(queries, chroma_collection)
        
        # Deduplicate the retrieved documents 去重，这一步很关键
        unique_documents = set()
        for documents in retrievedChunksList:
            for document in documents:
                unique_documents.add(document)

        # Generate course content and visualize it.
        courseContent = write_one_lesson(
            client, 
            lesson_description, 
            list(unique_documents), 
            language, 
            style_options, 
            ts_suggestions,
            model
        )
        ss.messages_ui.append({"role": "assistant", "content": decorate_suggested_questions_assistant(count_generating_content, ss.language, list_augmentedQueries)})
    return courseContent

def regenerate_outline(course_outline_list):
    try:
        course_outline_string = ''
        lessons_count = 0
        for outline in course_outline_list:
            lessons_count += 1
            course_outline_string += f"**{lessons_count}. {outline[0]}**"
            course_outline_string += f"\n\n{outline[1]} \n\n"
        write_course_outline = st.expander("Check the course outline", expanded=False)
        with write_course_outline:
            st.markdown(course_outline_string)
    except Exception:
        display_general_warning()
        pass

def regenerate_content(course_content_list):
    try:
        count_generating_content = 0
        for content in course_content_list:
            count_generating_content += 1
            with st.expander(f"Learn the lesson {count_generating_content} ", expanded=False):
                st.markdown(content)
    except Exception:
        display_general_warning()
        pass

def display_main_page(is_visualized):
    description_1 = """
            :grey[Github Repo:] https://github.com/Siyuan-Harry/OmniTutor_2

            ### ✨ Key features                                           
                                                    
            - 🧑‍🏫 **Customized course creation**: <font color = 'grey'>Generated from any learning materials (**.md or .pdf**)!</font>
            - 📚 **All disciplines**: <font color = 'grey'>Whether it's math, physics, literature, history or coding, OmniTutor covers it all.</font>
            - ⚡️ **Turn any learning into a problem-solving game**: <font color = 'grey'>Based on the materials provided, OmniTutor continuously inspire you to dive deeper.</font>
            - 🔮 **You learn, OmniTutor learns too**: <font color = 'grey'>A multi-agent system (teacher, TA, and a teaching supervisor) is desgined to learn your status and habits.</font>
            """
    description_2 = """
    ### 🏃‍♂️ Get started!

    1. **Config basic info**: <font color = 'grey'>Give OmniTutor your own OpenAI API key and upload your learning materials.</font>
    """

    description_3 = """
    2. **Customize your learning journey**: <font color = 'grey'>Tell OmniTutor your preferred teaching style, topic, lesson count and language. All of these are optional.</font>
    """

    description_4 = """
    3. **Start course generating**: <font color = 'grey'>Touch "Start Learning" button below, then watch how OmniTutor creates personal-customized course for you.</font>
    """

    main_page = st.empty()
    if is_visualized:
        with main_page.container():
            st.markdown(description_1, unsafe_allow_html=True)

            st.markdown(description_2, unsafe_allow_html=True)
            api_key = st.text_input('🔑 Your OpenAI API key:', type="password", placeholder='sk-...')
            use_35 = st.checkbox('Use GPT-3.5 (GPT-4 is default)')
            added_files = st.file_uploader('📁 Upload .md or .pdf files, simultaneous mixed upload both types is supported.', type=['.md','.pdf'], accept_multiple_files=True)

            st.markdown(description_3, unsafe_allow_html=True)
            with st.expander('⚙️ Customize my course'):
                num_lessons = st.slider('How many lessons do you want this course to have?', min_value=2, max_value=15, value=5, step=1)
                custom_options = st.multiselect(
                    'Preferred teaching style :grey[(Recommend new users not to select)]',
                    ['More examples', 'More excercises', 'Easier to learn'],
                    max_selections = 2
                )
                learner_input = st.text_input('(Optional) Please enter what you want to learn 👇, you can enter keywords or phrases, and this may help OmniTutor create better lessons for you :)', 
                                              placeholder = 'Use comma "," to split the different keywords/phrases.')
                ss.language = 'English'
                Chinese = st.checkbox('Output in Chinese')
            
            st.markdown(description_4, unsafe_allow_html=True)
            btn_start = st.button('🔮 Start Learning')
        return api_key, use_35, added_files, num_lessons, custom_options, learner_input, Chinese, btn_start
    else:
        main_page.empty()
        return None

def display_chatInput_box(is_visualized):
    user_question = st.empty()
    if is_visualized:
        user_question = user_question.chat_input("Enter your questions when learning...")
    else:
        user_question.empty()

    return user_question


def display_current_status_col1():
    if ss.course_outline_list == []:
        if ss.chroma_collection != '':
            st.success('Processing file...Done')
            st.success("Constructing vector database from provided materials...Done")
        else:
            ss.main_page_displayed = True
    elif ss.course_outline_list != [] and ss.course_content_list == []:
        regenerate_outline(ss.course_outline_list)
    else:
        regenerate_outline(ss.course_outline_list)
        regenerate_content(ss.course_content_list)
    
def display_current_status_col2():
    st.caption(''':blue[AI Assistant]: Ask this TA any questions related to this course and get direct answers. :sunglasses:''')
    with st.chat_message("assistant"):
        st.markdown("Hello👋, how can I help you today? 😄")
    if ss.messages_ui != []:
        for message in ss.messages_ui:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        pass

def display_current_status():
    if ss.start_learning == 0:
        ss.main_page_displayed = False
    elif ss.start_learning == 1:
        display_main_page(False)
        col1, col2 = st.columns([0.6,0.4])
        with col1:
            display_current_status_col1()
        with col2:
            display_current_status_col2()

def display_warning_not_started():
    warning_not_started = st.empty()
    warning_not_started.warning("The learning haven't started yet, please start to learn first!", icon="⚠️")
    time.sleep(2)
    warning_not_started.empty()

def display_warning_started():
    general_warning = st.empty()
    general_warning.markdown("The learning has already started, please touch 'Next lesson' button!", icon="⚠️")
    time.sleep(2)
    general_warning.empty()

def display_warning_api_key():
    warning_api_key = st.empty()
    warning_api_key.markdown("🤯 请先输入正确的OpenAI API Key令牌 Please enter the OpenAI API Key first.")
    time.sleep(2)
    warning_api_key.empty()

def display_warning_upload_materials():
    warning_upload_materials = st.empty()
    warning_upload_materials.markdown("🤯 Please upload your file(s) first.")
    time.sleep(2)
    warning_upload_materials.empty()

def display_warning_upload_materials_vdb():
    warning_upload_materials_vdb = st.empty()
    warning_upload_materials_vdb.markdown('🤯 Please upload your learning material(s) and wait for constructing vector database first.')
    time.sleep(2)
    warning_upload_materials_vdb.empty()

def display_general_warning():
    general_warning = st.empty()
    general_warning.markdown('🤯Oops.. We encountered an error. Please try again.')
    time.sleep(2)
    general_warning.empty()

def convert_markdown_string(course_outline_list, course_content_list):
    course_markdown_string = ''
    lessons_count = 0
    for outline in course_outline_list:
        lessons_count += 1
        course_markdown_string += f"**{lessons_count}. {outline[0]}**"
        course_markdown_string += f"\n\n{outline[1]} \n\n"
    lessons_count = 0
    for content in course_content_list:
        lessons_count += 1
        course_markdown_string += f"# Lesson {lessons_count}\n\n"
        course_markdown_string += f"{content}\n\n"
    
    course_markdown_string += f"# 🦉 Chat History \n\n"
    for message in ss.messages_ui:
        if message["role"] == "user":
            course_markdown_string += f"- Me: \n\n{message['content']} \n\n"
        elif message["role"] == "assistant":
            course_markdown_string += f"- Assistant: \n\n{message['content']} \n\n"
    return course_markdown_string

