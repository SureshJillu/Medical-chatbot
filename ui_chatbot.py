import chainlit as cl
import asyncio
from medical_chatbot import get_response_from_medical_chatbot  

# Store conversation history
conversation_history = []

@cl.on_chat_start
async def on_chat_start():
    """Initialize chatbot session"""
    await cl.Message(content="Hello! I'm your medical assistant. How can I help you today?").send()


@cl.on_message
async def on_message(msg: cl.Message):
    user_input = msg.content

    # Send loading indicator message
    loading_msg = await cl.Message(content="⏳ Processing your query...").send()

    try:
        # Show "thinking" message
        thinking_msg = cl.Message(content="🤔 *Analyzing your question...*\n")
        await thinking_msg.send()
        
        # Fetch response
        response_text = get_response_from_medical_chatbot(user_input)

        # Ensure response is a string
        if not isinstance(response_text, str):
            response_text = str(response_text)

        # Send response in one go (instead of word-by-word updates)
        # Step 3: Send the actual response in one go
        response_msg = cl.Message(content="💡 **Response:**\n\n" + response_text)
        await response_msg.send()


        # Remove the loading message
        await loading_msg.remove()

    except Exception as e:
        await cl.Message(content="⚠️ An error occurred while processing your request.").send()
        print(f"Error: {e}")
