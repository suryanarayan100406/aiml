# 🧠 ANI Creative Flow Optimizer — User Manual

Welcome to **ANI**, your personal AI cognitive state classifier and flow optimizer. ANI is designed to monitor your work habits in real-time and help you achieve and maintain **Deep Work**.

This guide explains how to use the ANI dashboard and Chrome extension to track your cognitive flow.

---

## 🚀 1. Starting a Focus Session

When you open the ANI application, you'll start on the Dashboard. To begin tracking your flow state:

1. Click on **Session** in the left sidebar.
2. **Describe your task**: In the text box, write out what you are currently working on (e.g., *"Writing an email to the marketing team"* or *"Debugging the authentication module"*). ANI uses Natural Language Processing to instantly classify how cognitively demanding this task is.
3. **Enable Inputs**:
   - Click **Enable Microphone** to allow ANI to analyze your speech patterns (Words Per Minute, pauses, and vocal fluency).
   - Click **Share Screen** to allow ANI to analyze your visual distractions (tab counts, phone presence).
4. Click **Start Session**.

---

## 📊 2. Understanding Your Dashboard

Once a session is active, navigate back to the **Dashboard** to see your real-time cognitive metrics.

### The Flow State Ring
At the center of your dashboard is the Current Flow State ring. It will transition colors based on your current cognitive load:
* 🟣 **Deep Flow**: You are highly focused, undistracted, and working on a demanding task.
* 🟢 **Soft Flow**: You are focused, but the task is less demanding or slightly routine.
* 🟡 **Distracted**: You have too many tabs open, or you are looking at your phone.
* 🟠 **Task-Switching**: You are bouncing rapidly between different contexts.
* 🔴 **Pseudo-Working**: You are at your computer but not engaged in productive work (e.g., long pauses in speech, idle behavior).

### Modality Cards
Surrounding the Flow State ring are individual AI analysis cards:
* **👁️ Vision**: Shows how many browser tabs are visible, whether your phone is detected, and your overall visual focus ratio.
* **🎙️ Audio**: Displays your speaking speed (WPM) and vocal fluency. Erratic or overly slow speech can imply high cognitive strain or distraction.
* **📝 NLP**: Shows what type of work ANI thinks you are doing based on your initial task description, along with its estimated cognitive demand.

---

## 🧩 3. The Chrome Extension

ANI learns from *you*. The Chrome Extension passively collects data to fine-tune its understanding of what "Deep Work" looks like for your specific work style.

1. **Passive Tracking**: As you work, the extension silently counts how many tabs you have open and classifies the type of websites you visit.
2. **Self-Reporting Focus**: 
   - Click the **ANI icon** in your browser toolbar anytime during your workday.
   - You will see a quick popup asking: *"How focused are you right now?"*
   - Select a score from **1 (Distracted)** to **5 (Deep Flow)**.
   - Enter a brief note about what you are doing, then click **Submit Report**.
   
This self-reported data is saved to your local machine and is used by the AI to align its predictions with your actual feelings of productivity.

---

## ⚙️ 4. Reviewing Your Models
Curious about how the AI works? Click on the **Models** tab in the sidebar to see the live technical specifications of the 4 neural networks powering your experience (YOLOv8 for vision, XGBoost for Audio, DistilBERT for Text, and Random Forest for fusion).

---
*ANI respects your privacy. All audio, vision, and text processing happens directly on your machine. No screen captures or voice recordings are ever sent to the cloud.*
