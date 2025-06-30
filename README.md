<!--<h3><b>Black & White Photo Colorization using AI</b></h3>-->
## <b>Photo Colorization</b>

### Project Overview
Black & White Photo Colorization using AI!

Old black-and-white photos often carry valuable memories, but they can feel dull due to the lack of color. This project brings those photos to life by automatically colorizing them using artificial intelligence (AI). The model used is SIGGRAPH 2017 Colorization Model by Richard Zhang et al., integrated into an interactive Streamlit app.

This project demonstrates the application of all the Python skills learned throughout the course:
1. Modular programming & custom functions
2. Use of external libraries like torch, PIL, skimage
3. Streamlit for UI
4. Interactive elements: file upload, preview, dynamic result, download

### Key Features
1. Upload black and white photos (JPG, PNG)
2. Automatic colorization using pre-trained AI model (SIGGRAPH17)
3. Download the colorized image
4. Side-by-side Before & After comparison
5. Uses pre-trained model weights stored locally
6. Modular structure: siggraph17.py, base_color.py, utils.py, app.py

![image](https://github.com/user-attachments/assets/ab358470-9b07-4eb6-8a37-3864c32c5ed8)

**Clone the repository**

```
git clone https://github.com/duaribusatu/final_project_colorize-picture_REAID.git
cd final_project_colorize-picture_REAID
```

**Create and activate virtual environment**
```
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

**Install dependencies**
```
pip install -r requirements.txt
```

**Run the app**
```
streamlit run app.py
```

### App Preview
1. Upload ur image
![image](https://github.com/user-attachments/assets/b81432d4-3f20-4d78-83dc-42843b119f7a)

2. Colorize image
![image](https://github.com/user-attachments/assets/abe43247-4523-49ca-b73f-304416c9d985)
