"""
Transformer Explainer for Event Extraction by Lukas Künig

Diploma Thesis: Explainability of Transformer models for Event Extraction Applications in Industrial Maintenance

TU Wien, Institut of Management Science - Research Group of Smart and Knowledge-Based-Maintenance

Overview:
---------
This program provides a GUI interface to help users understand and interpret the results of a Transformer model for
event extraction. The program uses the Hugging Face library to load pretrained models and tokenizers, and SHAP for
explaining the model's predictions.

Features:
---------
1. Allows users to load a text file and a list of events.
2. Extracts events from the provided text using a custom transformer pipeline.
3. Displays the average event probability and quantity of extracted events using a bar plot.
4. Provides a detailed explanation for each sentence in the text using SHAP values.
5. Allows users to navigate through SHAP explanations sentence by sentence.

Main Components:
----------------
1. EventExtractionPipeline: A custom pipeline based on Hugging Face's ZeroShotClassificationPipeline, tailored for
event extraction.
2. process_text_and_labels: Processes the provided text and labels and returns the extracted events, their scores,
and SHAP values.
3. results: Generates a bar plot displaying the average probability and quantity of extracted events.
4. shap_explainer_per_sentence: Displays SHAP explanations for each detected event in a given sentence.
5. GUI components to facilitate user interaction, file loading, and result visualization.

Note: Ensure that the required libraries are installed and available before running the program.
Required libraries: matplotlib, numpy, pandas, sentencepiece, shap, tensorflow, torch, transformers

"""

# Imports
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import shap
import string
import numpy as np
import matplotlib.pyplot as plt
import nltk
from collections import defaultdict, Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer, ZeroShotClassificationPipeline

# Stopwords list
from nltk.corpus import stopwords
stop_words_english = set(stopwords.words('english'))
stop_words_german = set(stopwords.words('german'))
STOP_WORDS = stop_words_english.union(stop_words_german)


MODEL_NAME = 'MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli'
count_clicks = 0

# Load Hugging Face model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Creation of individual EventExtractionPipeline, based on the HuggingFaceZeroShotClassificationPipeline
class EventExtractionPipeline(ZeroShotClassificationPipeline):
    def __call__(self, *args):
        call_super = super().__call__(args[0], self.new_labels)[0]
        return [[{'label': x[0], 'score': x[1]} for x in zip(call_super['labels'], call_super['scores'])]]

    def set_new_labels(self, labels):
        self.new_labels = labels

def remove_stop_words(text, stop_words):
    # Split the text into words
    words = text.split()
    # Filter out the stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Join the filtered words back into a string
    filtered_text = ' '.join(filtered_words)
    return filtered_text

# Processing of text and labels
def process_text_and_labels(stored_text, stored_events_list):
    # Update the model's label2id and id2label configurations based on the given labels
    model.config.label2id.update({v: k for k, v in enumerate(stored_events_list)})
    model.config.id2label.update({k: v for k, v in enumerate(stored_events_list)})

    # Create an instance of the EventExtractionPipeline with the defined HuggingFace model and tokenizer
    event_extraction_pipe = EventExtractionPipeline(model=model, tokenizer=tokenizer)
    event_extraction_pipe.set_new_labels(stored_events_list)


    # Split the example text by periods to get individual sentences and clean them
    temp = [s.strip() + '.' for s in stored_text.split('.') if s]

    # Lists to store events and their associated scores for all sentences
    event_list_all = []
    score_list_all = []

    # Process each sentence to predict events
    for sentence in temp:
        prediction = event_extraction_pipe([sentence])
        event_score_list = prediction[0]

        # Extract scores and determine the threshold score for an event
        scores = [p['score'] for p in event_score_list]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        event_score = mean_score + (0.4 * std_score)

        # Calculate SHAP values for explainability
        sentence_without_punctuation = sentence.translate(str.maketrans('', '', string.punctuation))
        explainer = shap.Explainer(event_extraction_pipe)
        shap_values = explainer([sentence_without_punctuation])

        # Extract events with scores above the threshold
        for i in event_score_list:
            if i['score'] > event_score:
                event_list_all.append(i['label'])
                score_list_all.append(i['score'])

    # Returns events and scores with the SHAP values above the threshold
    return event_list_all, score_list_all, shap_values

# Function which receives as input a text and a list of events. The function returns a plot which returns a plot with
# a bar chart for event probability and a line chart for event quantity
def results(stored_text, stored_events_list):
    # Close any existing plots to ensure a fresh start
    plt.close('all')

    # Process the text and labels to get events, scores, and shap_values
    event_list_all, score_list_all, _ = process_text_and_labels(stored_text, stored_events_list)

    # Create a dictionary with events as keys and their scores as a list of values
    dict_from_lists = defaultdict(list)
    for k, v in zip(event_list_all, score_list_all):
        dict_from_lists[k].append(v)

    # Compute the average score for each event and store it in a new dictionary
    dic_from_list_average = {k: np.mean(v) for k, v in dict_from_lists.items()}

    # Extract the names of the events and their average scores
    names = list(dic_from_list_average.keys())
    values = list(dic_from_list_average.values())

    # Count the occurrence of each event in the list
    count_dict = Counter(event_list_all)
    frequency = list(count_dict.values())

    # Set up a dual Y-axis plot: one for average event probability and the other for event quantity
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Events')
    ax1.set_ylabel('Average Event Probability', color='blue')
    ax1.bar(names, values, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # The second Y-axis for event quantity
    ax2 = ax1.twinx()
    ax2.set_ylabel('Event Quantity', color='red')
    ax2.plot(names, frequency, color='red', marker='o')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_yticks(range(0, max(frequency) + 1))

    # Set general settings for the plot
    plt.title('Average Event Probability and Quantity')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.4)
    ax1.set_xticklabels(names, rotation=90)

    # Returns the plot
    return fig


# Function which receives as input one sentence and a list of events. The function returns a plot which returns a bar
# plot with the SHAP values of each word for every event detected in that sentence.
def shap_explainer_per_sentence(stored_events_list, current_sentence, original_sentence):
    # Close any existing plots to ensure a fresh start
    plt.close('all')

    # Process the current sentence and events to get events, scores, and shap_values
    event_list_all, score_list_all, shap_values = process_text_and_labels(current_sentence, stored_events_list)

    # List to store figures for each detected event
    figures = []
    for event_name, event_score in zip(event_list_all, score_list_all):

        # Extract the index of the current event from the stored events list
        index = stored_events_list.index(event_name)
        shap_values_for_event = np.array(shap_values.values)[:, :, index][0]

        # Process and combine SHAP values for visualization (Combine tokens from sentencepiece tokenizer into whole
        # words that humans can understand)
        x = shap_values.data[0]
        y = shap_values_for_event
        x_new = []
        y_new = []
        for i in range(len(x)):
            if x[i] != '':
                x_new.append(x[i])
                y_new.append(y[i])

        x_new_combined, y_new_combined = [], []
        temp_str, temp_val = "", 0

        for i, item in enumerate(x_new):
            item = item.replace(' ', '')
            if item[0] == '▁':
                if temp_str:
                    x_new_combined.append(temp_str[1:])
                    y_new_combined.append(temp_val)
                temp_str = item
                temp_val = y_new[i]
            else:
                temp_str += item
                temp_val += y_new[i]

        if temp_str:
            x_new_combined.append(temp_str[1:])
            y_new_combined.append(temp_val)

        # Generate a bar plot for SHAP values of each word in the sentence
        colors = ['green' if value > 0 else 'red' for value in y_new_combined]
        positions = range(len(x_new_combined))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(positions, y_new_combined, color=colors)
        ax.set_xticks(positions)
        ax.set_xticklabels(x_new_combined, rotation=90, fontsize=9)
        ax.set_ylabel('SHAP value of each word', fontsize=9)
        ax.set_title(f'Event: {event_name}') #(Pred.: {event_score:.2%})')
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.3)
        figures.append(fig)
        plt.close(fig)

    # Create a new GUI window to display the SHAP values for each sentence and event
    new_shap_window = tk.Toplevel(main_window)
    new_shap_window.title('SHAP Values per Sentence and Event')
    new_shap_window.geometry('1200x800')
    new_shap_window.resizable(False, False)

    # Display the title and details of the current sentence and detected events
    shap_title_label = tk.Label(new_shap_window, text='Spotted Events per sentence', font=('Arial', 25))
    shap_title_label.pack(pady=10)
    shap_label_0 = tk.Label(new_shap_window, text=f'Original Sentence: {original_sentence}', font=('Arial', 15))
    shap_label_0.pack(pady=10)
    shap_label_1 = tk.Label(new_shap_window, text=f'List of events in this sentence: {", ".join(event_list_all)}',
                            font=('Arial', 15))
    shap_label_1.pack(pady=10)

    # Setup a canvas with a scrollbar to display SHAP diagrams
    canvas = tk.Canvas(new_shap_window)
    scrollbar = tk.Scrollbar(new_shap_window, orient='vertical', command=canvas.yview)
    frame = tk.Frame(canvas)
    frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.create_window((0, 0), window=frame, anchor='center')
    canvas.configure(yscrollcommand=scrollbar.set)

    # A frame to hold the individual SHAP diagrams
    inner_frame = tk.Frame(canvas)
    inner_frame.pack(fill=tk.BOTH, expand=True)

    # Embed each SHAP diagram in the GUI
    for fig in figures:
        fig_canvas = FigureCanvasTkAgg(fig, master=inner_frame)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=False, pady=30, padx=100)

    canvas.create_window((0, 0), window=inner_frame, anchor='nw')
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side='left', fill='both', expand=True)
    new_shap_window.mainloop()

# Opens a new window displaying a plot showing the probability and quantity of extracted events.
def open_plot_window(stored_text, stored_events_list):
    fig = results(stored_text, stored_events_list)

    # Create a new window to display the plot
    new_window = tk.Toplevel(main_window)
    new_window.title('Result')
    result_title_label = tk.Label(new_window, text='Overview Event Results', font=('Arial', 25))
    result_title_label.pack(pady=10)
    result_label_1 = tk.Label(new_window,
                              text='The average event probability and the event quantity of the extracted events from '
                                   'the whole text are shown in the diagram.',
                              font=('Arial', 15))
    result_label_1.pack(pady=10)
    new_window.geometry('800x600')

    # Add the plot to the window using a canvas
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Retrieves the content from text widgets, processes it, and opens a window to display the results.
def store_as_variables(text_content, event_content_label):
    stored_text = remove_stop_words(text_content.get(1.0, tk.END).strip(), STOP_WORDS)
    stored_events = event_content_label.get(1.0, tk.END).strip()
    stored_events_list = [element.strip() for element in stored_events.split(',')]
    open_plot_window(stored_text, stored_events_list)

# Processes the content from text widgets, updates a button's text, and runs the SHAP explainer for sentences.
def store_as_variables_run_shap(text_content, event_content_label, shap_button):
    global count_clicks
    count_clicks += 1
    original_text = text_content.get(1.0, tk.END).strip()
    stored_text = remove_stop_words(original_text, STOP_WORDS)
    stored_events = event_content_label.get(1.0, tk.END).strip()
    stored_events_list = [element.strip() for element in stored_events.split(',')]
    stored_text_list = [s.strip() + '.' for s in stored_text.split('.')]
    original_text_list = [s.strip() + '.' for s in original_text.split('.')]
    if count_clicks < len(stored_text_list):
        store_button_shap.config(text='Plot SHAP values of the next sentence')
        current_sentence = stored_text_list[count_clicks - 1]
        original_sentence = original_text_list[count_clicks - 1]
        shap_explainer_per_sentence(stored_events_list, current_sentence, original_sentence)
    if count_clicks == len(stored_text_list):
        store_button_shap.config(text='Text finished!')

# Main-GUI-Setup
if __name__ == "__main__":
    # Initialize the main GUI window
    main_window = tk.Tk()
    main_window.title('TEEE_by_Künig')
    main_window.geometry('600x700')
    main_window.resizable(False, False)

    # Function to close the main GUI window
    def exit_program():
        main_window.destroy()

    # Create a menu bar for the main GUI window
    menu_bar = tk.Menu(main_window)
    main_window.config(menu=menu_bar)

    # Add an option in the menu bar to close the program
    file_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Exit", command=exit_program)

    # Set the main title for the GUI window
    title_label = tk.Label(main_window, text='Transformer Event Extraction Explainer', font=('Arial', 25))
    title_label.pack(pady=10)

    # GUI setup for event input: label, text widget, and button to load events from a file
    event_label = tk.Label(main_window, text='Insert your Events:')
    event_button = tk.Button(main_window, text='Browse Events', command=lambda: load_event_file(event_content_label))
    event_content_label = tk.Text(main_window, height=5, width=60, borderwidth=2, relief='solid')
    event_label.pack()
    event_content_label.pack(pady=10)
    event_button.pack()

    # GUI setup for text input: label, text widget, and button to load text from a file
    text_label = tk.Label(main_window, text='Insert your Text:')
    text_button = tk.Button(main_window, text='Browse Text', command=lambda: load_text_file(text_content))
    text_content = tk.Text(main_window, height=20, width=60, borderwidth=2, relief='solid')
    text_label.pack(pady=10)
    text_content.pack(pady=10)
    text_button.pack()

    # Button to visualize the results overview
    store_button_overview = tk.Button(main_window, text='Event Results Overview',
                                      command=lambda: store_as_variables(text_content, event_content_label))
    store_button_overview.pack(pady=10)

    # Button to visualize the SHAP values for the first sentence
    store_button_shap = tk.Button(main_window, text='SHAP Values of the first sentence',
                                  command=lambda: store_as_variables_run_shap(text_content, event_content_label,
                                                                              store_button_shap))
    store_button_shap.pack()

    # Start the main loop of the GUI
    main_window.mainloop()
