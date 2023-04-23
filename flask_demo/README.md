# Flask Demo
## Environment Setup
- It is recommended to use an environment to run this demo (if you have created an environment previously to run the models in this repo, the same can be used here)
- Install dependencies from `requirements.txt`

## Commands used:
1. To run demo: ```python main.py```
  - the port used here is **9000**, feel free to change it to another port of your choice.
2. On a browser, navigate to `http://localhost:9000/` to start the demo


## Webpage description:
1. Home Page: this is a simple page which accepts images to be uploaded from the user
![Home page](https://github.com/TheLohia/Phteven/blob/master/flask_demo/screenshots/Home%20Page.png)

  - Select Image: user can upload an image (standard image formats only)
![Select Image](https://github.com/TheLohia/Phteven/blob/master/flask_demo/screenshots/Select%20image.png)

2. Result Page: the result of the model prediction is displayed here
![Result page](https://github.com/TheLohia/Phteven/blob/master/flask_demo/screenshots/Result%20Page.png)

## Debug info:
- If the demo fails to start, it is likely that port 9000 is currently being used by another program, please change the port in the `port` in the `app.run` method
