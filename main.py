from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

# Initialisieren des Modells und der Eingabeaufforderung
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)

# Definieren Sie die Kette, um Eingabeaufforderung und Modell zu kombinieren
chain = prompt | model

def handle_conversation():
    context = ""
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Aufruf des Modells mit Kontext und Frage
        result = chain.invoke({"context": context, "question": user_input})
        print("Bot: ", result)
        
        # Aktualisieren des Kontexts mit der aktuellen Unterhaltung
        context += f"\nUser: {user_input}\nAI: {result}"

if __name__ == "__main__":
    handle_conversation()
