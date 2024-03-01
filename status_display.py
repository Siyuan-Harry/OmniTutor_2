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
    if "messages_ui" not in ss:
        ss.messages_ui = []
    if "messages" not in ss:
        ss.messages = []

    if "num_lessons" not in ss:
        ss.num_lessons = ''
    if "language" not in ss:
        ss.language = ''
    if "style_options" not in ss:
        ss.style_options = ''

    if "start_learning" not in ss:
        ss.start_learning = 0
    if "is_displaying_description" not in ss:
        ss.is_displaying_description = 0

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

def display_current_status_col1(write_description, description):
    if ss.course_outline_list == []:
        if ss.chroma_collection != '':
            write_description.markdown(description, unsafe_allow_html=True)
            st.success('Processing file...Done')
            st.success("Constructing vector database from provided materials...Done")
        else:
            write_description.markdown(description, unsafe_allow_html=True)
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

def display_current_status(write_description, description):
    if ss.start_learning == 0:
        write_description.markdown(description)
    elif ss.start_learning == 1:
        write_description.empty()
        col1, col2 = st.columns([0.6,0.4])
        with col1:
            display_current_status_col1(write_description, description)
        with col2:
            display_current_status_col2()

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