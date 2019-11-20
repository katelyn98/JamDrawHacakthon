# Dare Mighty Things Hackathon & Conference 2019
### JamDraw - using computer vision, machine learning, and data science

---
**Correct file for drive is named as FinalProjDriver.py**
' Visual Demonstration coming soon '

We created a web application that takes your stories (drawings) and generates a song based off the percentage of a dominant color in your drawing. 

To create this, we started off with rapid prototyping harnessing the design thinking skills. This lead us to come up with such a unique idea: JamDraw. 

Our web application is run through a flask front end and uses OpenCV (Computer Vision Library) to detect and track an object. It uses Pandas to filter through spotify song data to find the attribute "danceability" for a song. 

We also incorporated machine learning using an unsupervised learning method: KMeans Clustering. KMeans Clustering was used to determine the dominant color in the image by narrowing down to three clusters and creating a histogram of the ratio of the three most dominant colors in the image. A percentage was associated with the second color out of the three which then translated to a value on the danceability scale which is an attribute of a song through the Spotify API. 

