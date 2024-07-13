const backendUrl = "http://127.0.0.1:8000"; // Replace with your FastAPI backend URL

function handleUpload(endpoint) {
  const fileInput = endpoint === 'predict' ? document.getElementById('fileUpload1') : document.getElementById('fileUpload2');
  const resultDiv = endpoint === 'predict' ? 'predictionResult1' : 'predictionResult2';
  const errorDiv = endpoint === 'predict' ? 'error1' : 'error3';

  const file = fileInput.files[0];
  if (!file) {
    document.getElementById(errorDiv).innerText = "Please select a file to upload";
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  fetch(`${backendUrl}/${endpoint}/`, {
    method: 'POST',
    body: formData,
  })
  .then(response => {
    if (response.ok) {
      return response.json();
    } else {
      throw new Error(`Error ${response.status}: ${response.statusText}`);
    }
  })
  .then(data => {
    document.getElementById(resultDiv).innerHTML = `
      <h3>Prediction Result</h3>
      <p>Filename: ${data.filename}</p>
      <p>Probability: ${data.probability}</p>
      <p>Final Verdict: ${data['final verdict']}</p>
    `;
    document.getElementById(errorDiv).innerText = "";
  })
  .catch(error => {
    document.getElementById(errorDiv).innerText = `Error uploading the file: ${error.message}`;
  });
}

function getServerImage() {
  fetch(`${backendUrl}/get_image`)
  .then(response => {
    if (response.ok) {
      return response.blob();
    } else {
      throw new Error(`Error ${response.status}: ${response.statusText}`);
    }
  })
  .then(imageBlob => {
    const imageUrl = URL.createObjectURL(imageBlob);
    document.getElementById('serverImage').innerHTML = `<img src="${imageUrl}" alt="Server Image">`;
    document.getElementById('error2').innerText = "";
  })
  .catch(error => {
    document.getElementById('error2').innerText = `Error fetching server image: ${error.message}`;
  });
}
