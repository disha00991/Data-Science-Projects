# Color Palette Creation
![Python](https://img.shields.io/badge/Python-3.7.9-brightgreen) ![sklearn](https://img.shields.io/badge/sklearn-library-yellowgreen.svg) ![Pandas](https://img.shields.io/badge/pandas-library-green.svg) ![Seaborn](https://img.shields.io/badge/seaborn-library-orange.svg)

## Project Overview
* This project uses one of the many Clustering Algorithms - chosen by the user, to create clusters of colors present in an image and generates a Color Palette consisting of a user defined number of colors.
* A flask app deployed with heroku. Checkout the app here: https://color-clustering.herokuapp.com/

![demo](https://user-images.githubusercontent.com/13835601/128101864-b9563df1-3cc8-4c91-a83e-fbff89af275c.mp4)

## Inspiration:
- This idea initially came to my mind when I saw a beautiful aurora scenery and wanted to paint it but only wanted to buy minimum colors for the purpose. I thought, what if Clustering could quickly tell me which colors most occur in the painting based on the number of colors I want to buy (the number of clusters that get created).
<img src="readme_resources/result3.png" width=300/>
- Later that day, I noticed that while we create Instagram stories, it always picks colors itself for a background behind our video/image. Some of my own stories and their beautiful backgrounds selected by Instagram AI led me to think how this is being done!
<p float="left">
<img src="readme_resources/img1.png" width=200/>
<img src="readme_resources/img2.png" width=200/>
<img src="readme_resources/img3.png" width=200/>
</p>
- One way that this can be done is a quick clustering and sample a few top most colors from the palette thus created.
- For both the above use cases, I perform clustering for any given image url using Machine Learning clustering algorithms.

## Clustering Algorithms:
I used Heirarchical clustering, Kmeans, Kmeans++ and Birch algorithms and noticed they generate similar results. As such even a single algorithm does not produce similar clustering when run twice!

## Some Results:
This project when deployed to heroku usually times out as the clustering algorithms take very long to output the clusters! But the project runs smooth on local machine. Some of the interesting palettes the I created:
<p float="left">
<img src="readme_resources/result1.png" width=300/>
<img src="readme_resources/result2.png" width=300/>
<img src="readme_resources/result3.png" width=300/>
</p>

## Next steps:
This project when deployed to heroku usually times out as the clustering algorithms take very long to output the clusters! So now, I am trying to reduce this time taken. For now, using low resolution images helps to yield outputs fast on the deployed version!

_**----- Important Note -----**_<br />
• If you encounter this webapp as shown in the picture given below, it is occuring just because **free dynos for this particular month provided by Heroku have been completely used.** _You can access the webpage on 1st of the next month._<br />
• Sorry for the inconvenience.

![Heroku-Error](readme_resources/application-error-heroku.png)
