# 🧹 Messy Desk Scanner

A Streamlit web app that uses YOLO object detection to scan your desk via webcam, detect clutter, and rate its messiness on a "Chaos Score" scale.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://messy-desk-scanner.streamlit.app/)

## 🔍 What It Does

- Captures a photo using your **webcam**
- Uses **YOLOv8** to detect clutter like books, bottles, mugs, trash, etc.
- Calculates a **chaos score** based on object types
- Displays a **visual detection overlay**
- Gives a fun, descriptive **messiness verdict**

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt

