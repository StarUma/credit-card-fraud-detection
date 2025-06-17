document.getElementById("prediction-form").addEventListener("submit", async function (event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);
    let jsonData = {};

    for (let [key, value] of formData.entries()) {
        if (value.trim() === "") continue;  // Skip empty fields
        jsonData[key] = parseFloat(value);
    }

    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(jsonData)
    });

    const result = await response.json();
    document.getElementById("result").innerText = "Prediction: " + result.prediction;
});
