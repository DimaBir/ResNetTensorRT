from flask import render_template, request, redirect, url_for
from app import app
from app.utils import init, run_benchmark, run_prediction


@app.route('/demo', methods=['GET', 'POST'])
def demo():
    if request.method == 'POST':
        # Initialize the model and inference classes
        initialization_data = init()

        # Handle file upload and model selection
        image = request.files['image']
        operation = request.form['operation']

        # Save the uploaded image
        image_path = os.path.join('app/static/images/uploaded_images', image.filename)
        image.save(image_path)

        if operation == 'prediction':
            results = run_prediction(image_path, initialization_data)
            return render_template('results.html', results=results, image_path=image_path)

        elif operation == 'benchmark':
            plot_path = run_benchmark(initialization_data)
            return render_template('results.html', plot_path=plot_path)

    return render_template('demo.html')
