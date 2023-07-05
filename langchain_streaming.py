from langchain.chat_models import ChatOpenAI
from langchain.schema import (
  HumanMessage, )
import openai

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import config

openai_api_key = config.api_keys['openai_api_image_key']
openai.api_key = openai_api_key
template ='''
Use the following pieces of context from waitbutwhy to answer the question at the end. 
If you don't know the answer, just clarify that you are not sure, but this might be how Tim Urban thinks.
'''
store = '''
fatty tuna on the right (all from the same fish). Interestingly, in the doc, Jiro says that the fattier pieces are simple and predictable and though most people like them the most, it’s actually the lean tuna where the subtle sophisticated flavor specific to that particular tuna comes out. I won’t pretend that I could discern the extra sophistication going on when I had it, but it was very good (the medium was my favorite piece of the whole meal). A few other notes: The atmosphere was tense . Everyone was incredibly polite with me, but they were short with each other. Jiro’s son would curtly whisper something under his breath to his second-in-command, who would in turn quietly bark something to guy under him, who kept getting yelled at and was having a horrible time. They would often request “no soy” for certain pieces, i.e. “Don’t put soy sauce on this bite of perfection you foreign fool.” I had a chat with the second-in-command guy for a bit, who spoke decent English. When I asked
to things like stir frying vegetables and rice, which is an excuse to have a condiment/spices party. As for cookbooks, I recently got J. Kenji Lopez-Alt’s The Food Lab , which is delightful so far. Not a big chef fanboy person, although I had a phase with Bobby Flay’s Boy Meets Grill show at one point and I’m pretty infatuated with how cocky and rad Danny Bowien is (the Mission Chinese guy). My greatest culinary passion is whichever 15 hot sauces I happen to have in my fridge at any given time. Currently: ___________ How does wind work? What is the mechanism behind wind happening? Why does it happen? – Steven H. (Cape Girardeau, MO) I’ll need to read about how weather works for 30 hours before I can give a proper answer to this question, but I’ll take a crack. The key to wind is air pressure . Forget the atmosphere for a second and take a breath in. What happens? Air from outside your body rushes into your body. Now exhale—and air rushes back out. What’s happening is that the air
stir fries, but there are a million different kind of Japanese foods and it’s one of those countries where everything is good. It didn’t matter what I ordered or where, it was just all good. Even the convenience store meals: 5) The sex industry is everywhere and spans a wide range with confusing borders From what I saw on the streets and learned by talking to a few expats about it, the range seems to look something like this: The Standard Stuff region includes the brothels, shady massage parlors, and strip clubs that line the streets (especially in certain areas of town), the large number of sex shops, and some super-weird institutions like blowjob bars and vibrator bars (I don’t know what either of these is exactly, but they both exist). When I expressed my surprise to an expat that a huge sex industry would exist in what seemed like such an otherwise buttoned-down culture, I was told that it’s more complicated than the sex industry in another country—that there’s a very
Lid For any of you readers considering creating a new, tense nation of ethnic and religious groups who don’t like each other
'''
user_input = '''Write a post about soy sauce within 300 words like waitbutwhy using "you" in a casual language'''

messages = [{"role": "system", "content": template + store}, 
            {"role": "user", "content":user_input}]
response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
print("response", response)

prompt = "Write me a song about sparkling water."

original_message = [{"role": "user", "content": prompt}]
def stream_chat(prompt):
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    ):
        print("chunk", chunk)
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            yield content
# stream = stream_chat(prompt)
# print(next(stream))
# print(next(stream))
# print(next(stream))






def stream_langchain_chat(): 
    chat = ChatOpenAI(streaming=True,
                      temperature=0, 
                      openai_api_key=openai_api_key)
    for chunk in chat([HumanMessage(content="Write me a song about sparkling water.")]): 
        print("chunk", chunk)
    # print(chat.content)
    # print("position 0", chat[0])
    # print("position 1", chat[1])
        # print(chunk)
        # chunk.get("delta", ()).get("content")