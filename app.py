from raw_functions import *
from status_display import *
from openai import OpenAI
import streamlit as st

def app():
    initialize_session_state()

    with st.sidebar:
        api_key = st.text_input('🔑 Your OpenAI API key:', 'sk-...')
        use_35 = st.checkbox('Use GPT-3.5 (GPT-4 is default)')
        st.image("https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/WechatIMG1729.jpg")
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
    
    st.title("OmniTutor 2.0")
    st.subheader("Your personalized :blue[AI Knowledge Engine] 🦉")
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
    write_description = st.empty()
    write_description.markdown(description, unsafe_allow_html=True)
    
    btn_start = st.button('Start Learning')

    user_question = st.chat_input("Enter your questions when learning...")

    #displaying current status
    if ss.start_learning == 1:
        display_current_status(write_description, description)

    if btn_start:
        write_description.empty()
        if api_key !="" and api_key.startswith("sk-") and len(api_key) == 51 and added_files:
            ss.start_learning = 1
            ss.num_lessons = num_lessons
            ss.style_options = add_prompt_course_style(custom_options)
            if ss["OPENAI_API_KEY"] == '':
                ss["OPENAI_API_KEY"] = api_key
                st.success("✅ API Key stored successfully!")
            if Chinese:
                ss.language = "Chinese"
            if use_35:
                ss["openai_model"] = 'gpt-3.5-turbo-1106'
            client = OpenAI(api_key = ss["OPENAI_API_KEY"])

            col1, col2 = st.columns([0.6,0.4])
            with col1:
                if len(ss["OPENAI_API_KEY"]) != 51 and added_files:
                    display_warning_api_key()
                    display_current_status(
                        write_description, 
                        description, 
                    )
                elif not added_files:
                    write_description.empty()
                    display_warning_upload_materials()
                    write_description.markdown(description, unsafe_allow_html=True)
                else:
                    ss.temp_file_paths = initialize_file(added_files)
                    ss.chroma_collection = initialize_vdb(ss.temp_file_paths)
                    ss.course_outline_list = initialize_outline(client, ss.temp_file_paths, num_lessons, ss.language, ss["openai_model"])
                    btn_next = st.button('Okay, next learning step! ⏩️')

    if btn_next:
        write_description.empty()
        if api_key !="" and api_key.startswith("sk-") and len(api_key) == 51 and added_files:
            col1, col2 = st.columns([0.6,0.4])
            with col1:
                if ss.course_outline_list != [] and ss.course_content_list == []:
                    regenerate_outline(ss.course_outline_list)
                    ss.lesson_counter = 1
                    new_lesson = visualize_new_content(
                        client, 
                        ss.lesson_counter, 
                        ss.course_outline_list[ss.lesson_counter-1], 
                        ss.chroma_collection, 
                        ss.language, 
                        ss.style_options, 
                        ss["openai_model"]
                    )
                    ss.course_content_list.append(new_lesson)
                    btn_next = st.button('Okay, next learning step! ⏩️')
                else:
                    if ss.lesson_counter < ss.num_lessons:
                        regenerate_outline(ss.course_outline_list)
                        regenerate_content(ss.course_content_list)
                        ss.lesson_counter += 1
                        new_lesson = visualize_new_content(
                            client,
                            ss.lesson_counter,
                            ss.course_outline_list[ss.lesson_counter-1],
                            ss.chroma_collection,
                            ss.language, 
                            ss.style_options, 
                            ss["openai_model"]
                        )
                        ss.course_content_list.append(new_lesson)
                        btn_next = st.button('Okay, next learning step! ⏩️')
                    elif ss.lesson_counter >= ss.num_lessons:
                        display_current_status_col1(write_description, description)
                        #让用户下载课程的文稿markdown
                        course_md = convert_markdown_string(ss.course_outline_list,ss.course_content_list)
                        st.download_button(
                            label="Download Course Script",
                            data=course_md,
                            file_name='OmniTutor_Your_Course.md',
                        )
                        btn_next = st.button('Okay, next learning step! ⏩️')
            with col2:
                display_current_status_col2()


    if user_question:
        write_description.empty()
        if len(ss["OPENAI_API_KEY"]) != 51:
            display_warning_api_key()
            display_current_status(
                write_description, 
                description, 
            )
        elif ss["OPENAI_API_KEY"] != '' and ss.chroma_collection == '':
            display_warning_upload_materials_vdb()
            display_current_status(
                write_description, 
                description, 
            )
        else:
            client = OpenAI(api_key = ss["OPENAI_API_KEY"])
            col1, col2 = st.columns([0.6,0.4])
            with col1:
                display_current_status_col1(write_description, description)
            with col2:
                st.caption(''':blue[AI Assistant]: Ask this TA any questions related to this course and get direct answers. :sunglasses:''')

                with st.chat_message("assistant"):
                    st.markdown("Hello👋, how can I help you today? 😄")

                # Display chat messages from history on app rerun
                for message in ss.messages_ui:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                #更新ui上显示的聊天记录
                ss.messages_ui.append({"role": "user", "content": user_question})
                # Display new user question.
                with st.chat_message("user"):
                    st.markdown(user_question)

                retrieved_chunks_for_user = searchVDB(user_question, ss.chroma_collection)
                prompt = decorate_user_question(user_question, retrieved_chunks_for_user)
                ss.messages.append({"role": "user", "content": prompt})

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    full_response = get_visualize_stream_completion_from_messages(
                        client,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages #用chatbot那边的隐藏消息记录
                        ],
                        model=ss["openai_model"]
                    )
                ss.messages.append({"role": "assistant", "content": full_response})
                ss.messages_ui.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    app()

    