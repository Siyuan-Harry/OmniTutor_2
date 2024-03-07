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
        ss["openai_model"] = 'gpt-4-1106-preview' 
    if "client" not in ss:
        ss.client = ''
    if "messages_ui" not in ss:
        ss.messages_ui = []
    if "messages" not in ss:
        ss.messages = []

    if "num_lessons" not in ss:
        ss.num_lessons = 0
    if "language" not in ss:
        ss.language = ''
    if "style_options" not in ss:
        ss.style_options = ''

    if "start_learning" not in ss:
        ss.start_learning = 0
    if "main_page_displayed" not in ss:
        ss.main_page_displayed = True

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

def initialize_vdb(temp_file_paths):
    with st.spinner('Constructing vector database from provided materials...'):
        chroma_collection = constructVDB(temp_file_paths)
    st.success("Constructing vector database from provided materials...Done")
    return chroma_collection

def initialize_outline(client, temp_file_paths, num_lessons, language, model):
    with st.spinner('Generating Course Outline...'):
        summarized_materials = get_keywords(temp_file_paths)
        course_outline_list = genarating_outline(client, summarized_materials, num_lessons, language, model)
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

def visualize_new_content(client, count_generating_content, lesson_description, chroma_collection, language, style_options, model):
    retrievedChunksList = searchVDB(lesson_description, chroma_collection)
    with st.expander(f"Learn the lesson {count_generating_content} ", expanded=False):
        courseContent = write_one_lesson(
            client, 
            lesson_description, 
            retrievedChunksList, 
            language, 
            style_options, 
            model
        )
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
    description = """
            <p style = "color: grey;"> An all-round teacher. A teaching assistant who really knows the subject. **Anything. Anywhere. All at once.** </p> :100:
            
            Github Repo: https://github.com/Siyuan-Harry/OmniTutor_2
            - Github Repo (for OmniTutor prototype version): https://github.com/Siyuan-Harry/OmniTutor 

            ### ✨ Key features                                           
                                                    
            - 🧑‍🏫 **Concise and clear course creation**: <font color = 'grey'>Generated from your learning notes (**.md**) or any learning materials (**.pdf**)!</font>
            - 📚 **All disciplines**: <font color = 'grey'>Whether it's math, physics, literature, history or coding, OmniTutor covers it all.</font>
            - ⚙️ **Customize your own course**: <font color = 'grey'>Choose your preferred teaching style, lesson count and language.</font>
            - ⚡️ **Fast respond with trustable accuracy**: <font color = 'grey'>Problem-solving chat with the AI teaching assistant who really understand the materials.</font>
            
            ### 🏃‍♂️ Get started!
                                                        
            1. **Input Your OpenAI API Key**: <font color = 'grey'>Give OmniTutor your own OpenAI API key (On top of the **sidebar**) to get started.</font>
            2. **Upload learning materials**: <font color = 'grey'>The upload widget in the sidebar supports PDF and .md files simutaenously.</font>
            3. **Customize your course**: <font color = 'grey'>By few clicks and swipes, adjusting teaching style, lesson count and language for your course.</font>
            4. **Start course generating**: <font color = 'grey'>Touch "Next Leaning Step!" button in the sidebar, then watch how OmniTutor creates personal-customized course for you.</font>
            5. **Interactive course generation**: <font color = 'grey'>Whenever you finish one leaning step, ouch "Next Leaning Step!" button to continue. You will never be left behind.</font>
            6. **Interactive learning**: <font color = 'grey'>Ask OmniTutor any questions related to this course whenever you encountered them.</font>
                                    
            ###### 🎉 Have fun playing with Omnitutor!                                                                                                              
            """
    main_page = st.empty()
    if is_visualized:
        with main_page.container():
            st.markdown(description, unsafe_allow_html=True)
            btn_start = st.button('Start learning!')
            api_key = st.text_input('🔑 Your OpenAI API key:', 'sk-...')
            use_35 = st.checkbox('Use GPT-3.5 (GPT-4 is default)')
            added_files = st.file_uploader('📁 Upload .md or .pdf files, simultaneous mixed upload both types is supported.', type=['.md','.pdf'], accept_multiple_files=True)
            with st.expander('⚙️ Customize my course'):
                num_lessons = st.slider('How many lessons do you want this course to have?', min_value=2, max_value=15, value=5, step=1)
                custom_options = st.multiselect(
                    'Preferred teaching style :grey[(Recommend new users not to select)]',
                    ['More examples', 'More excercises', 'Easier to learn'],
                    max_selections = 2
                )
                ss.language = 'English'
                Chinese = st.checkbox('Output in Chinese')
        return api_key, use_35, added_files, num_lessons, custom_options, Chinese, btn_start
    else:
        main_page.empty()
        return None

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
    warning_not_started.warning("The learning haven't stated yet, please start to learn first!", icon="⚠️")
    time.sleep(2)
    warning_not_started.empty()

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

    return course_markdown_string

