import tkinter as tk
from .quiz_data import get_questions
from tkinter import messagebox, StringVar

import tkinter as tk
from tkinter import messagebox, StringVar, Frame, Canvas, Scrollbar

class QuizApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Quiz Application")
        self.root.attributes('-fullscreen', True)  # Make the window full screen
        
        self.questions = get_questions()
        
        self.current_question_index = None
        self.selected_answer = StringVar(value = "1")
        self.groups = []
        self.scores = {}
        self.answered_questions = set()
        self.current_group_index = 0
        
        self.create_initial_interface()
    
    def create_initial_interface(self):
        tk.Label(self.root, text="Enter number of groups:").grid(row=0, column=0, padx=10, pady=10)
        self.num_groups_entry = tk.Entry(self.root)
        self.num_groups_entry.grid(row=0, column=1, padx=10, pady=10)
        
        tk.Button(self.root, text="Submit", command=self.setup_groups).grid(row=1, column=0, columnspan=2, padx=10, pady=10)
    
    def setup_groups(self):
        try:
            num_groups = int(self.num_groups_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number")
            return
        
        if num_groups < 1:
            messagebox.showerror("Input Error", "Number of groups must be at least 1")
            return
        
        for widget in self.root.winfo_children():
            widget.grid_forget()
        
        self.group_name_entries = []
        for i in range(num_groups):
            tk.Label(self.root, text=f"Enter name for group {i + 1}:").grid(row=i, column=0, padx=10, pady=5)
            group_name_entry = tk.Entry(self.root)
            group_name_entry.grid(row=i, column=1, padx=10, pady=5)
            self.group_name_entries.append(group_name_entry)
        
        tk.Button(self.root, text="Start Quiz", command=self.start_quiz).grid(row=num_groups, column=0, columnspan=2, padx=10, pady=10)
    
    def start_quiz(self):
        self.scores = {entry.get(): 0 for entry in self.group_name_entries if entry.get()}
        if len(self.scores) != len(self.group_name_entries):
            messagebox.showerror("Input Error", "All groups must have a unique name")
            return
        
        for widget in self.root.winfo_children():
            widget.grid_forget()
        
        self.create_quiz_interface()
        self.update_scoreboard()
        self.update_current_group_label()
        self.show_question_buttons()
    
    def create_quiz_interface(self):
        # Scoreboard on the left
        self.scoreboard_frame = tk.LabelFrame(self.root, text="Scoreboard", padx=10, pady=10)
        self.scoreboard_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)
        
        # Question buttons in the middle
        self.question_button_canvas = Canvas(self.root, width=400, height=600)
        self.question_button_canvas.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.scrollbar = Scrollbar(self.root, orient="vertical", command=self.question_button_canvas.yview)
        self.scrollbar.grid(row=0, column=2, sticky='ns')
        
        self.question_button_frame = Frame(self.question_button_canvas)
        self.question_button_frame.bind(
            "<Configure>",
            lambda e: self.question_button_canvas.configure(
                scrollregion=self.question_button_canvas.bbox("all")
            )
        )
        
        self.question_button_canvas.create_window((0, 0), window=self.question_button_frame, anchor="nw")
        self.question_button_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.question_buttons = []
        for i in range(len(self.questions)):
            btn = tk.Button(self.question_button_frame, text=f"Question {i + 1}", 
                            command=lambda i=i: self.choose_question(i))
            btn.grid(row=i//4, column=i%4, padx=5, pady=5)
            self.question_buttons.append(btn)
        
        # Question and options on the right
        self.quiz_frame = tk.Frame(self.root)
        self.quiz_frame.grid(row=0, column=3, sticky="nsew", padx=10, pady=10)
        self.root.grid_columnconfigure(3, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        self.current_group_var = StringVar()
        self.current_group_label = tk.Label(self.quiz_frame, textvariable=self.current_group_var, font=("Arial", 18))
        self.current_group_label.grid(row=0, column=0, pady=10, columnspan=2)
        
        self.question_var = StringVar()
        self.question_label = tk.Label(self.quiz_frame, textvariable=self.question_var, font=("Arial", 14), wraplength=600)
        self.question_label.grid(row=1, column=0, pady=20, columnspan=2)
        
        self.option_vars = []
        self.option_buttons = []
        for i in range(4):
            var = StringVar()
            self.option_vars.append(var)
            button = tk.Radiobutton(self.quiz_frame, variable=self.selected_answer, value=var.get())
            button.grid(row=2+i, column=0, sticky="w", columnspan=2)
            self.option_buttons.append(button)

        self.submit_button = tk.Button(self.quiz_frame, text="Submit Answer", command=self.submit_answer)
        self.submit_button.grid(row=6, column=0, pady=10, columnspan=2)
        
        self.end_quiz_button = tk.Button(self.root, text="End Quiz", command=self.end_quiz)
        self.end_quiz_button.grid(row=1, column=3, pady=10, padx=10, sticky='se')
    
    def show_question_buttons(self):
        self.question_label.grid_forget()
        for button in self.option_buttons:
            button.grid_forget()
        self.submit_button.grid_forget()
        
        for i, btn in enumerate(self.question_buttons):
            if i in self.answered_questions:
                btn.config(state=tk.DISABLED)
            else:
                btn.config(state=tk.NORMAL)
    
    def show_question(self):
        self.question_label.grid(row=1, column=0, pady=20, columnspan=2)
        for i, button in enumerate(self.option_buttons):
            button.grid(row=2 + i, column=0, sticky="w", columnspan=2)
        self.submit_button.grid(row=6, column=0, pady=10, columnspan=2)
    
    def update_scoreboard(self):
        for widget in self.scoreboard_frame.winfo_children():
            widget.destroy()
        
        sorted_scores = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        for group, score in sorted_scores:
            tk.Label(self.scoreboard_frame, text=f"{group}: {score}").pack(anchor='w')
    
    def update_current_group_label(self):
        group_name = list(self.scores.keys())[self.current_group_index % len(self.scores)]
        self.current_group_var.set(f"Current Group: {group_name}")
    
    def choose_question(self, question_index):
        self.current_question_index = question_index
        self.question_buttons[question_index].config(state=tk.DISABLED)
        
        question_data = self.questions[question_index]
        self.question_var.set(question_data["question"])
        for i, option in enumerate(question_data["options"]):
            self.option_vars[i].set(option)
            self.option_buttons[i].config(text=option, value=option)
        
        self.show_question()
    
    def submit_answer(self):
        if self.current_question_index is None:
            messagebox.showwarning("No Question", "Please choose a question first")
            return
        
        selected = self.selected_answer.get()
        correct_answer = self.questions[self.current_question_index]["answer"]
        
        group_name = list(self.scores.keys())[self.current_group_index % len(self.scores)]
        
        if selected == correct_answer:
            self.scores[group_name] += 10
            messagebox.showinfo("Correct Answer", f"The correct answer is: {correct_answer}\n\n\
{self.questions[self.current_question_index].get('explanation')} ")
        else:
            messagebox.showinfo("Incorrect Answer", f"The correct answer is: {correct_answer}\n\n\
{self.questions[self.current_question_index].get('explanation')} ")
        
        self.answered_questions.add(self.current_question_index)
        self.update_scoreboard()
        
        self.current_group_index += 1
        self.update_current_group_label()
        
        if len(self.answered_questions) == len(self.questions):
            messagebox.showinfo("Quiz Finished", "All questions have been answered.")
            self.end_quiz()
        else:
            self.clear_question()
            self.show_question_buttons()

    def clear_question(self):
        self.current_question_index = None
        self.question_var.set("")  # Clear the question text
        self.selected_answer.set(value = "1")  # Clear the selected answer
        for button in self.option_buttons:
            button.deselect()  # Deselect all radio buttons

    
    def end_quiz(self):
        for widget in self.root.winfo_children():
            widget.grid_forget()

        # Display final scores and close button
        final_scores_frame = tk.Frame(self.root)
        final_scores_frame.grid(row=0, column=0, sticky="nsew")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        scores_label = tk.Label(final_scores_frame, text="Final Scores", font=("Arial", 18))
        scores_label.grid(row=0, column=0, pady=10)

        sorted_scores = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        for i, (group, score) in enumerate(sorted_scores, start=1):
            score_label = tk.Label(final_scores_frame, text=f"{group}: {score}", font=("Arial", 14))
            score_label.grid(row=i, column=0, pady=5)

        close_button = tk.Button(final_scores_frame, text="Close", command=self.root.quit)
        close_button.grid(row=len(self.scores) + 1, column=0, pady=20)