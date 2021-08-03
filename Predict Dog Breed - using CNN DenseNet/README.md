# CS 532 21S - Project 2: ML Inference
ml-inference-team125g created by GitHub Classroom

In this project we implement a simple image classification server. We have used a pre-trained densenet121 model, wrapped it in a flask application and built a Docker container to use the model for inference.

### To run project with docker:
- Download the docker image of the project named *“532_project”* here: https://bit.ly/33ueKOI
- After downloading, run this command to make a local docker image:
  *docker load -i 532_project.image*
- Run the docker image using the command(do not forget the dot in the end):
  *docker run -p 5000:5000 532_project .*
- Use the command (replace <images/dog.jpg> with path of the image you want to predict relative to the current directory of command prompt): 
  *curl -F “file=@images/dog.jpg” http:127.0.0.1:5000/predict*
- Alternatively, you can change the image path in the *run_me.bat* file in our project directory and double click on it to run the inference for your own image. (The path of the image should be relative to the current directory of command prompt)


### To run project without docker:
- Clone the repository and in the project directory run: *python app.py*
- Then for getting a dummy prediction about an image in images/dog.jpg simply double click on the file: *run_me.bat*
- At this point, your result should be *“Samoyed”*
- Alternatively, use the command (replace <images/dog.jpg> with path of the image you want to predict relative to the current directory of command prompt): 
  *curl -F “file=@images/dog.jpg” http:127.0.0.1:5000/predict*
  
  
  @authors Disha Singh | Pranjali Ajay Parse
