let probChart = null; // Global variable to hold the chart instance

document.getElementById('image-form').addEventListener('submit', function(e) {
    e.preventDefault();
    let formData = new FormData(this);
    let submitButton = document.querySelector("#image-form button[type='submit']");

    // Disable the submit button and show the spinner
    submitButton.disabled = true;
    document.getElementById('spinner').style.display = 'block';

    // Display the mini image
    let imageInput = document.getElementById('image');
    if (imageInput.files && imageInput.files[0]) {
        let reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('processedImage').src = e.target.result;
        };
        reader.readAsDataURL(imageInput.files[0]);
    }

    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data)
        // Enable the submit button and hide the spinner
        submitButton.disabled = false;
        document.getElementById('spinner').style.display = 'none';

        if (data.predictions) {
            displayPredictions(data.predictions);
        } else if (data.benchmark) {
            displayBenchmark(data.benchmark);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        // Enable the submit button in case of an error
        submitButton.disabled = false;
        document.getElementById('spinner').style.display = 'none';
    });
});

function displayPredictions(predictions) {
    const processedImageContainer = document.getElementById('processedImageContainer');
    const probGraphContainer = document.getElementById('probGraphContainer');

    processedImageContainer.style.display = 'block';
    probGraphContainer.style.display = 'block';

    // Display the mini image
    let imageInput = document.getElementById('image');
    if (imageInput.files && imageInput.files[0]) {
        let reader = new FileReader();
        reader.onload = function(e) {
            let processedImage = document.getElementById('processedImage');
            if (processedImage) {
                processedImage.src = e.target.result;
                processedImage.style.maxWidth = '450px'; // Adjust width as needed
                processedImage.style.height = 'auto'; // Maintain aspect ratio
            }
        };
        reader.readAsDataURL(imageInput.files[0]);
    }

    // Render prediction probabilities graph
    renderProbGraph(predictions);
}

function renderProbGraph(predictions) {
    const ctx = document.getElementById('probGraph').getContext('2d');

    // Destroy the existing chart if it exists
    if (probChart) {
        probChart.destroy();
    }

    const labels = predictions.map(prediction => prediction.label);
    const probs = predictions.map(prediction => (prediction.confidence * 100).toFixed(2)); // Convert to percentage
    const backgroundColors = predictions.map(() => `rgba(${randomRGB()}, ${randomRGB()}, ${randomRGB()}, 0.45)`); // Random colors

    probChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: probs,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors.map(color => color.replace('0.2', '1')), // Darker border color
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y', // Set to 'y' for horizontal bars
            scales: {
                x: {
                    beginAtZero: true
                }
            }
        }
    });
}

function displayBenchmark(benchmarkResults) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = ''; // Clear previous results

    for (const model in benchmarkResults) {
        const time = benchmarkResults[model].avgTime;
        const throughput = benchmarkResults[model].avgThroughput;

        const p = document.createElement('p');
        p.textContent = `${model} - Average Time: ${time.toFixed(2)} ms, Throughput: ${throughput.toFixed(2)}`;
        resultsDiv.appendChild(p);
    }

    // If you have data for plotting (e.g., for 'ALL' mode), call displayLineGraph
    if (benchmarkResults['all']) {
        displayLineGraph(benchmarkResults['all']);
    }
}

function randomRGB() {
    return Math.floor(Math.random() * 255);
}

function displayBenchmark(benchmarkResults) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = ''; // Clear previous results

    for (const model in benchmarkResults) {
        const time = benchmarkResults[model].avgTime;
        const throughput = benchmarkResults[model].avgThroughput;

        const p = document.createElement('p');
        p.textContent = `${model} - Average Time: ${time.toFixed(2)} ms, Throughput: ${throughput.toFixed(2)}`;
        resultsDiv.appendChild(p);
    }

    // If you have data for plotting (e.g., for 'ALL' mode), call displayLineGraph
    if (benchmarkResults['all']) {
        displayLineGraph(benchmarkResults['all']);
    }
}

function displayLineGraph(data) {
    document.getElementById('lineGraphContainer').style.display = 'block';

    const ctx = document.getElementById('lineGraph').getContext('2d');
    const labels = Object.keys(data);
    const times = labels.map(label => data[label].time);
    const throughputs = labels.map(label => data[label].throughput);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Inference Time (ms)',
                data: times,
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1,
                yAxisID: 'y-axis-time',
            }, {
                label: 'Throughput',
                data: throughputs,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
                yAxisID: 'y-axis-throughput',
            }]
        },
        options: {
            scales: {
                'y-axis-time': {
                    type: 'linear',
                    display: true,
                    position: 'left',
                },
                'y-axis-throughput': {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    });
}
