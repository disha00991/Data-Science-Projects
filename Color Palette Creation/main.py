# Importing essential libraries
from flask import Flask, request, render_template
from clustering_algos import Clustering_algos
from PIL import Image
from io import BytesIO
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/color_palette', methods=['POST'])
def color_palette():
    algo = request.form['algo']
    url = request.form['url']
    n_colors = request.form['n_colors']
    if request.method == 'POST':
        img = Image.open(BytesIO(requests.get(url).content))

        cluster_obj = Clustering_algos(img, algo, int(n_colors))
        palette = cluster_obj.get_palette_plot()

        def convert_to_rgb_string(color_arr):
            return 'rgb('+ str(color_arr[0]) +','+ str(color_arr[1]) +','+ str(color_arr[2]) + ')'

        palette = list(map(convert_to_rgb_string, palette))
        response = {
            'palette': palette,
            'image_url': url
        }
        return render_template('index.html', response=response)

# curl command
#>curl -X POST -d "is_url=true" -d "url=https://d21zeai4l2a92w.cloudfront.net/wp-content/uploads/2020/01/ColorChangingFlowers.jpg" http://127.0.0.1:5000/get_color_palette --output palet.png
