import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import re
from tkinter import *
import contractions
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')

# Load the trained machine learning model
RF = joblib.load('Fake_News_Detection_using_NLP.pkl')

def preprocess_text(x):
    cleaned_text = re.sub(r'[^a-zA-Z\d\s\']+', '', x)
    word_list = []
    for each_word in cleaned_text.split(' '):
        try:
            word_list.append(contractions.fix(each_word).lower())
        except:
            print(x)
    return " ".join(word_list)

count_vectorizer = CountVectorizer(max_features=200,ngram_range=(1, 2))
tf_idf_transformer = TfidfTransformer(smooth_idf=False)


def print_name():
    name = name_entry.get()  # Get the name from the input field
    output_label.config(text=f"Hello, {name}!")  # Display the name in a label

def output_lable(n):
    global classification
    if n == 0:
        classification = tk.Label(root, text="This is a Fake News", font=('helvetica', 10 , 'bold'), fg="red")
        classification.pack(anchor = 'center')
    elif n == 1:
        classification = tk.Label(root, text="This is a Genuine news", font=('helvetica', 10, 'bold'), fg="green")
        classification.pack(anchor = 'center')
        
def manual_testing():
    news = news_entry.get("1.0", "end-1c")
    #preprocessing data
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(preprocess_text)
    new_x_test = new_def_test["text"]
    new_xv_test = count_vectorizer.fit_transform(new_x_test)
    new_xvw_test = tf_idf_transformer.fit_transform(new_xv_test)
    pred_RF = RF.predict(new_xvw_test)
    
    output_label.config(text = output_lable(pred_RF[0]))
    
def clear_entry():
    news_entry.delete('1.0', END)    
    classification.destroy()

# Create the main application window
root = tk.Tk()
root.title("Fake News Detection")
root.geometry('600x400')

# Create a label
instruction_label = tk.Label(root, text="Enter News Text:",font=('Comic Sans MS',15))
instruction_label.pack()

# Create an entry field to input the name
news_entry = tk.Text(root, height = 10, width=60,borderwidth=5,font=('Comic Sans MS',10))
news_entry.pack()


# Create a button to trigger the name printing
check_button = tk.Button(root, text = 'Check', font=('helvetica', 10 , 'bold'), fg = 'black', bg = 'white', command = manual_testing)
check_button.pack(padx=10,pady=10)

clear_button = tk.Button(root, text = 'Clear', font=('helvetica', 10 , 'bold'), fg = 'black', bg = 'white', command = clear_entry)
clear_button.pack(padx=20,pady=10)


# Create a label to display the printed name
output_label = tk.Label(root, text="",)
output_label.pack()


# Start the GUI event loop
root.mainloop()