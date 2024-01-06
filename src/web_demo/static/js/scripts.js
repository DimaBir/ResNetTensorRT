let probChart = null; // Global variable to hold the chart instance

document.getElementById('image-form').addEventListener('submit', function(e) {
    e.preventDefault();
    let formData = new FormData(this);
    let submitButton = document.querySelector("#image-form button[type='submit']");
    let mode = document.getElementById('mode').value;
    let cnnModel = document.getElementById('cnnModel').value;

    // Hide the info card
    document.getElementById('info-card').style.display = 'none';

    // Hide benchmark graphs when switching to prediction mode
    if (mode === 'predict') {
        document.getElementById('timeGraphContainer').style.display = 'none';
        document.getElementById('throughputGraphContainer').style.display = 'none';
    } else {
        // Hide prediction elements when switching to benchmark mode
        document.getElementById('processedImageContainer').style.display = 'none';
        document.getElementById('probGraphContainer').style.display = 'none';

        // Start updating benchmark info
        updateBenchmarkInfo();
    }

    // Disable the submit button and show the spinner
    submitButton.disabled = true;
    document.getElementById('spinner').style.display = 'block';

    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.status === 400) {  // Check for rate limit exceeded
            throw new Error("File format invalid. Please use: JPG, Jpeg, PNG, Gif. Max 500MB");
        }
        if (response.status === 429) {  // Check for rate limit exceeded
            throw new Error("Rate limit exceeded. Please try again later.");
        }
        return response.json();
    })
    .then(data => {
        // Enable the submit button and hide the spinner
        submitButton.disabled = false;
        document.getElementById('spinner').style.display = 'none';
        document.getElementById('benchmarkInfo').style.display = 'none';

        if (data.predictions) {
            displayPredictions(data.predictions, data.inference_time);
        } else if (data.benchmark) {
            displayBenchmark(data.benchmark);
        }
    })
    .catch(error => {
        console.error('Error:', error);

        // Display error message to the user
        displayFlashMessage("danger", error.message);
        // Enable the submit button in case of an error
        submitButton.disabled = false;
        document.getElementById('spinner').style.display = 'none';
        document.getElementById('benchmarkInfo').style.display = 'none';
    });
});

function displayFlashMessage(category, message) {
    let container = document.querySelector('.container');

    // Create the flash message div
    let flashMessageDiv = document.createElement('div');
    flashMessageDiv.className = `alert alert-${category} flash-message`;
    flashMessageDiv.role = 'alert';

    // Create the message text
    let messageText = document.createElement('span');
    messageText.textContent = message;
    flashMessageDiv.appendChild(messageText);

    // Create the countdown bar
    let countdownBar = document.createElement('div');
    countdownBar.className = 'countdown-bar';
    flashMessageDiv.appendChild(countdownBar);

    // Create the close button
    let closeButton = document.createElement('button');
    closeButton.className = 'close-button';
    closeButton.innerHTML = '&times;';
    closeButton.onclick = function() {
        clearInterval(countdownInterval);
        flashMessageDiv.remove();
    };
    flashMessageDiv.appendChild(closeButton);

    // Insert the flash message into the container
    container.insertBefore(flashMessageDiv, container.firstChild);

    // Start the countdown
    let timeLeft = 5; // Duration in seconds
    let countdownInterval = setInterval(() => {
        countdownBar.style.width = `${(timeLeft / 5) * 100}%`;
        countdownBar.style.backgroundColor = `rgba(255, 0, 0, ${timeLeft / 5})`;
        if (timeLeft <= 0) {
            clearInterval(countdownInterval);
            fadeOutElement(flashMessageDiv, countdownBar);
        }
        timeLeft--;
    }, 1000);
}

function fadeOutElement(element, excludeElement) {
    var fadeEffect = setInterval(function () {
        if (!element.style.opacity) {
            element.style.opacity = 1;
        }
        if (element.style.opacity > 0) {
            element.style.opacity -= 0.1;
            if (excludeElement) {
                excludeElement.style.opacity = 1; // Keep the countdown bar fully visible
            }
        } else {
            clearInterval(fadeEffect);
            element.remove();
        }
    }, 100);
}


document.getElementById('mode').addEventListener('change', updateModelOptions);
updateModelOptions();

function displayPredictions(predictions, inferenceTime) {
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

    // Update the inference time container
    let inferenceTimeDiv = document.getElementById('inferenceTime');
    if (inferenceTimeDiv) {
        // Remove the element
        inferenceTimeDiv.remove();

        // Create a new div element for inference time
        let newInferenceTimeDiv = document.createElement('div');
        newInferenceTimeDiv.id = 'inferenceTime';
        newInferenceTimeDiv.className = 'inference-time-container';
        newInferenceTimeDiv.innerHTML = `Inference Time: ${inferenceTime.toFixed(2)} ms`;

        // Re-add the element to the DOM
        let probGraphContainer = document.getElementById('probGraphContainer');
        probGraphContainer.appendChild(newInferenceTimeDiv);
    }

    // Update the inference time container
    let cnnModel = document.getElementById('cnnModel').value;
    let topPrediction = document.getElementById('topPrediction');
    if (topPrediction) {
        // Remove the element
        topPrediction.remove();

        // Create a new div element for inference time
        let newTopPrediction = document.createElement('div');
        newTopPrediction.id = 'topPrediction';
        newTopPrediction.className = 'top-prediction-container';
        newTopPrediction.innerHTML = `${cnnModel.toUpperCase()} thinks it is: ${predictions[0].label}`;

        // Re-add the element to the DOM
        let probGraphContainer = document.getElementById('probGraphContainer');
        probGraphContainer.appendChild(newTopPrediction);
    }
}

function renderProbGraph(predictions) {
    const ctx = document.getElementById('probGraph').getContext('2d');

    // Destroy the existing chart if it exists
    if (probChart) {
        probChart.destroy();
    }

    const labels = predictions.map(prediction => prediction.label);
    const probs = predictions.map(prediction => (prediction.confidence * 100).toFixed(2)); // Convert to percentage

    // Define a blue color palette
    const bluePurplePalette = [
        'rgba(0, 123, 255, 0.8)', // Bootstrap primary blue
        'rgba(23, 162, 184, 0.8)', // A lighter shade of blue
        'rgba(40, 167, 69, 0.8)', // A greenish-blue shade
        'rgba(0, 105, 217, 0.8)', // A darker shade of blue
        'rgba(3, 169, 244, 0.8)'  // A sky blue shade
    ];

    // Assign colors from the palette to each bar
    const backgroundColors = probs.map((_, index) => bluePurplePalette[index % bluePurplePalette.length]);

    probChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: probs,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors.map(color => color.replace('0.8', '1')), // Darker border color
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

function randomRGB() {
    return Math.floor(Math.random() * 255);
}

function displayBenchmark(benchmarkResults) {
    console.log(benchmarkResults)
    // Hide prediction elements
    document.getElementById('processedImageContainer').style.display = 'none';
    document.getElementById('probGraphContainer').style.display = 'none';

    // Display benchmark graphs
    document.getElementById('timeGraphContainer').style.display = 'block';
    document.getElementById('throughputGraphContainer').style.display = 'block';

    // Prepare data for line graphs
    const labels = Object.keys(benchmarkResults);
    const times = labels.map(label => benchmarkResults[label][0]);
    const throughputs = labels.map(label => benchmarkResults[label][1]);

    // Display line graphs
    displayLineGraph(labels, times, throughputs);
}

function displayLineGraph(labels, times, throughputs) {
    const timeGraphContainer = document.getElementById('timeGraphContainer');
    const throughputGraphContainer = document.getElementById('throughputGraphContainer');

    // Colors for the benchmark graphs
    const timeGraphColor = 'rgba(0, 123, 255, 0.8)'; // Bootstrap primary blue for the time graph
    const throughputGraphColor = 'rgba(23, 162, 184, 0.8)'; // A lighter shade of blue for the throughput graph

    if (timeGraphContainer && throughputGraphContainer) {
        timeGraphContainer.style.display = 'block';
        throughputGraphContainer.style.display = 'block';

        // Inference Time Graph
        const timeCtx = document.getElementById('timeGraph').getContext('2d');
        new Chart(timeCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Average Inference Time (ms)',
                    data: times,
                    backgroundColor: timeGraphColor,
                    borderColor: timeGraphColor.replace('0.8', '1'),
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

        // Throughput Graph
        const throughputCtx = document.getElementById('throughputGraph').getContext('2d');
        new Chart(throughputCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Average Throughput (samples/sec)',
                    data: throughputs,
                    backgroundColor: throughputGraphColor,
                    borderColor: throughputGraphColor.replace('0.8', '1'),
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
    } else {
        console.error('Error: Graph containers not found');
    }
}

function updateModelOptions() {
    const modeSelect = document.getElementById('mode');
    const modelSelect = document.getElementById('inferenceMode');

    // Clear existing options
    modelSelect.innerHTML = '';

    if (modeSelect.value === 'predict') {
        // Options for 'Predict' mode
        const options = ['ov', 'pytorch', 'onnx'];
        options.forEach(opt => {
            let option = document.createElement('option');
            option.value = opt;
            option.text = opt.toUpperCase(); // Capitalize first letter
            modelSelect.appendChild(option);
        });
    } else if (modeSelect.value === 'benchmark') {
        // Only 'ALL' option for 'Benchmark' mode
        let option = document.createElement('option');
        option.value = 'all';
        option.text = 'ALL';
        modelSelect.appendChild(option);
    }
}

function updateBenchmarkInfo() {
    const sentences = [
        "Analyzing model performance...",
        "Running benchmarks on different models...",
        "Calculating average inference time...",
        "Evaluating throughput metrics..."
    ];
    let currentSentence = 0;
    let currentChar = 0;

    const animatedText = document.getElementById('animatedText');
    const benchmarkInfo = document.getElementById('benchmarkInfo');
    benchmarkInfo.style.display = 'block';

    function typeSentence() {
        if (currentChar < sentences[currentSentence].length) {
            animatedText.textContent += sentences[currentSentence].charAt(currentChar);
            currentChar++;
            setTimeout(typeSentence, 100); // Delay between each character
        } else {
            // Wait before starting the next sentence
            setTimeout(() => {
                currentSentence = (currentSentence + 1) % sentences.length;
                animatedText.textContent = ''; // Clear the text
                currentChar = 0; // Reset character position
                typeSentence(); // Start typing the next sentence
            }, 3000); // Delay between sentences
        }
    }

    typeSentence();
}

document.addEventListener('DOMContentLoaded', function() {
    var dropZone = document.getElementById('drop-zone');
    var fileInput = document.getElementById('image');
    var clearBtn = document.getElementById('clear-btn');

    dropZone.addEventListener('click', function() {
        fileInput.click();
    });

    fileInput.addEventListener('change', function(event) {
        if (event.target.files.length) {
            handleFiles(event.target.files);
            clearBtn.style.display = 'block'; // Show clear button
        }
    });

    dropZone.addEventListener('dragover', function(event) {
        event.preventDefault();
        dropZone.style.backgroundColor = '#f0f0f0'; // Visual feedback for drag over
    });

    dropZone.addEventListener('dragleave', function(event) {
        dropZone.style.backgroundColor = ''; // Reset background color when drag leaves
    });

    dropZone.addEventListener('drop', function(event) {
        event.preventDefault();
        dropZone.style.backgroundColor = ''; // Reset background color on drop
        if (event.dataTransfer.files.length) {
            fileInput.files = event.dataTransfer.files;
            handleFiles(event.dataTransfer.files);
            clearBtn.style.display = 'block'; // Show clear button
        }
    });

    clearBtn.addEventListener('click', function() {
        resetDropZone();
    });

    function handleFiles(files) {
        if (files.length > 0) {
            var file = files[0];
            console.log('File type: ', file.type);

            // Check file type
            var allowedTypes = ['image/jpeg', 'image/png', 'image/gif'];
            if (!allowedTypes.includes(file.type)) {
                alert('Error: File format invalid. Please use: JPG, JPEG, PNG, GIF.');
                return;
            }

            // Check file size (500MB in bytes)
            var maxSize = 500 * 1024 * 1024;
            if (file.size > maxSize) {
                alert('Error: File too large. Max 500MB.');
                return;
            }

            var reader = new FileReader();
            reader.onload = function(e) {
                var img = document.createElement('img');
                img.src = e.target.result;
                img.style.maxWidth = '200px'; // Set a max width for the preview image
                img.style.maxHeight = '200px'; // Set a max height for the preview image
                img.style.marginTop = '10px';
                dropZone.innerHTML = ''; // Clear the current content
                dropZone.appendChild(img); // Add the image to the drop zone
            };
            reader.readAsDataURL(file);
        }
    }

    function resetDropZone() {
        fileInput.value = ''; // Clear the file input
        dropZone.innerHTML = '<span>Drag and drop an image here or click to select a file</span>'; // Reset drop zone content
        clearBtn.style.display = 'none'; // Hide clear button
    }
});



