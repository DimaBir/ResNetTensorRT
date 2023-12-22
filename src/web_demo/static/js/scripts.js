document.getElementById('image-form').addEventListener('submit', function(e) {
    e.preventDefault();
    let formData = new FormData(this);

    document.getElementById('spinner').style.display = 'block';

    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data)
        document.getElementById('spinner').style.display = 'none';
        if (data.predictions) {
            displayPredictions(data.predictions);
        } else if (data.benchmark) {
            displayBenchmark(data.benchmark);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

function displayPredictions(predictions) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = ''; // Clear previous results

    predictions.forEach(prediction => {
        const p = document.createElement('p');
        p.textContent = `Label: ${prediction.label}, Confidence: ${prediction.confidence.toFixed(2)}`;
        resultsDiv.appendChild(p);
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
