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
    // Clear previous results
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    // Display the processed image
    let imageInput = document.getElementById('image');
    if (imageInput.files && imageInput.files[0]) {
        let reader = new FileReader();
        reader.onload = function(e) {
            let processedImage = document.getElementById('processedImage');
            processedImage.src = e.target.result;
            processedImage.style.width = '150px'; // Adjust width as needed
            processedImage.style.height = 'auto'; // Maintain aspect ratio

            // Display the div containing the processed image
            let processedImageDiv = document.getElementById('processedImageDiv');
            processedImageDiv.style.display = 'block';
        };
        reader.readAsDataURL(imageInput.files[0]);
    }

    // Prepare data for the probability graph
    const labels = predictions.map(p => p.label);
    const probs = predictions.map(p => p.confidence * 100); // Convert to percentage

    // Destroy the previous chart if it exists
    if (window.probChart) {
        window.probChart.destroy();
    }

    // Create a new chart with datalabels for percentages
    const ctx = document.getElementById('probGraph').getContext('2d');
    window.probChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: probs,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }]
            },
            plugins: {
                datalabels: {
                    color: '#000',
                    anchor: 'end',
                    align: 'top',
                    formatter: (value, context) => {
                        return value.toFixed(2) + '%'; // Format the label with percentage
                    }
                }
            }
        },
        plugins: [ChartDataLabels] // Ensure this plugin is included in your project
    });
}

// Function to generate random colors
function generateColors(count) {
    const colors = [];
    for (let i = 0; i < count; i++) {
        const r = Math.floor(Math.random() * 255);
        const g = Math.floor(Math.random() * 255);
        const b = Math.floor(Math.random() * 255);
        colors.push(`rgba(${r}, ${g}, ${b}, 0.5)`);
    }
    return colors;
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
