async function predict() {
  const text = document.getElementById("facts").value.trim();
  const resultDiv = document.getElementById("result");

  if (!text) {
    resultDiv.style.display = "block";
    resultDiv.className = "conf-low";
    resultDiv.textContent = "Please enter some facts.";
    return;
  }

  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ facts: text }),
  });

  const data = await response.json();

  let conf = data.confidence;

  // Color coding
  if (conf >= 70) resultDiv.className = "conf-high";
  else if (conf >= 40) resultDiv.className = "conf-mid";
  else resultDiv.className = "conf-low";

  resultDiv.style.display = "block";
  resultDiv.innerHTML = `
        <strong>Prediction:</strong> ${data.prediction} <br>
        <strong>Confidence:</strong> ${conf}%
    `;
}
