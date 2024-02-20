# OmniTutor 2.0

## Your personalized AI Knowledge Engine 🦉

![](pic2.jpg)

## app links: 

- **Prototype version (OmniTutor)**: https://huggingface.co/spaces/Siyuan0730/OmniTutor
- **This brand-new version (OmniTutor 2.0)**: https://omnitutor2-qjmzigr9qpyekfufsd2wjv.streamlit.app/

## Introduction

📢 **OmniTutor 2** is a fully updated version with greater robustness from the previous [OmniTutor prototype](https://github.com/Siyuan-Harry/OmniTutor/tree/main) version, and the reason for start this new git repo (rather than just create a new branch on old one) is, the old one is now used fully for teaching purpose. 

🧑‍🏫 OmniTutor is an all-round teacher, combined with a personalized teaching assistant that specially designed for all learners. Skilled in **creating customized courses** from given knowledge materials, and **answering course-related questions with trustable accuracy**. Hopefully it helps reducing obstacles in your learning journey.

## Features

- 🧑‍🏫 **Concise and clear course creation**: Generated from any learning materials (.md or .pdf)!
- 💡 **Removal of AI illusion**: visualization and controllability of knowledge processing process at every step.
- 📚 **All disciplines**: Whether it's math, physics, literature, history or coding, OmniTutor covers it all.
- ⚙️ **Customize your own course**: Choose preferred teaching style, lesson count and language.
- ⚡️ **Fast respond with trustable accuracy**: ask the AI teaching assistant who really understand the materials.

OmniTutor - An all-round teacher and a personalized teaching assistant who really knows the subject, to help you solve all your learning problems, Make learning so simple: anything, anywhere, all at once.

## How to Use

1. **Start the Application**: Execute the script to initiate the OmniTutor interface.
2. **Upload learning materials**: The upload widget in the sidebar supports PDF and .md files simutaenously.
3. **Customize your course**: By few clicks and swipes, adjusting teaching style, lesson count and language for your course.
4. **Start course generating**: Touch "Next learning step!" button in the sidebar, then watch how OmniTutor creates personal-customized course for you.
5. **Interactive learning**: Learn the course, and ask OmniTutor any questions related to this course whenever you encountered them.

## Setup and Running

Before running , ensure you have the following prerequisites installed:

- Python 3.x
- Streamlit
- FAISS
- NLTK
- OpenAI

1. **Clone the Repository**:

   ```bash
   git clone ...
   ```

2. **Install Required Libraries**:

   ```bash
   pip install -r requirements
   ```

3. **Set Up OpenAI API Key**:
   Ensure you have your OpenAI API key set up. You can either set it as an environment variable or use Streamlit's secrets management.

4. **Run the Application**:

   ```bash
   streamlit run <filename>.py
   ```

## 🎯 Future Plans

- [x] A button allows users to **download the generated course** freely!
- [x] **Generate each lesson with streaming output** to reduce annoying waiting
- [x] Ask questions as you learn
- [x] Add **session management system** for robustness
- [ ] **Update the session management system** to prevent the visual course content from disappearing in some cases
- [ ] **visualize chunks and RAG system** to enhance learners' trust and sense of control over the generated lessons
- [ ] Make the generated course outline **editable**


## Contributing

We welcome contributions from the community. If you'd like to improve the application, feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. For more details, refer to the `LICENSE` file in the repository.

## Acknowledgements

Thanks to the developers of Streamlit, FAISS, NLTK and OpenAI for their incredible tools that made this project possible.

Special thanks to Xuying Li, as my mentor and colleague, who gave me most significant support during the whole process of this project from zero to one. 
