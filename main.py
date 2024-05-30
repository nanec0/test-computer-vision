from tempfile import NamedTemporaryFile

import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool
import os
from shutil import copyfileobj
from PIL import Image
from dotenv import load_dotenv
import re
from tools import ImageProcessingHelper

load_dotenv()
# Path to the logo image
logo_path = "crdn.jpg"

##############################
### initialize agent #########
##############################
tools = [ImageCaptionTool()]

# Create an instance of the helper class
helper = ImageProcessingHelper()


conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=1,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key= os.getenv('OPENAI_API_KEY'),
    temperature=1,
    model_name="gpt-4o"
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    # max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)




# Open the image file
img = Image.open(logo_path)

# Resize the image to a fixed size
fixed_width = 300
fixed_height = 150
img = img.resize((fixed_width, fixed_height))

# Display the image
st.image(img, use_column_width=False)


# set header
st.header("Please upload an image")

# # set title
# st.title('Ask a question to an image')

# upload file
file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    # display image
    st.image(file, use_column_width=True)

    # text input
    # user_question = st.text_input('Ask a question about your image:')
    user_task = """
    Extract objects from the image caption. The objects are classified into three categories:
    - Textile Restoration: cloth, fabric, garment, textile, towel, blanket, clothing, shoe, handbag, leather, luggage, curtain, taxidermy, linen, rug, pillow, bedspread, comforter.
    - Electronic Restoration: laptop, phone, camera, electronic, TV, tablet, smartphone, computer, printer, monitor, speaker, console.
    - Art Restoration: painting, sculpture, art, picture, frame, print, drawing, poster, vase, china, lamp, photo, album, candle, mural, mosaic, doll, collection, coin, antique, clock, toy, memorabilia, tapestry, flag, textile art, book, manuscript, map, document, etching, statuette.
    If the object is similar to some of these examples, indicate the possible category and always give a confidence score in '%' from 0 to 100. If the confidence score is less than 60%, it will raise an alert for human validation.
    I want the response to be : Object: {object_name} \n Category: {category} \n Confidence: {confidence}%\n'
    """

    ##############################
    ### compute agent response ###
    ##############################
    # ensure the images directory exists
    if not os.path.exists('Images'):
        os.makedirs('Images')

    # save the file
    image_path = os.path.join('Images', file.name)
    with open(image_path, 'wb') as f:
        f.write(file.getvalue())

    # write agent response
    if user_task and user_task != "":
        with st.spinner(text="In progress..."):
            response = agent.run('{}, this is the image path: {}'.format(user_task, image_path))
            # Call process_objects
            print(response)
            # detections = helper.process_objects(response)
            st.write(response)



