window.sdpMatrixCollection1 = {};
window.sdpMatrixCollection2 = {};
window.sdpMatrixCollection3 = {};
window.sdpMatrixCollection4 = {};
window.sdpCurrentMatrixIndex = null;
let currentMatrix = '';

function updateSDPButtons() {
    const n = parseInt(document.getElementById('SDPRows').value, 10) || 2;
    const container = document.getElementById('SDP-matrix-buttons');
    container.innerHTML = '';
    for (let i = 0; i <= n; i++) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-secondary mr-2 mb-2';
        let label = '';
        if (currentMatrix === 'sdp-F_d') {
            label = `\\(F_{${i}}(\\delta)\\)`;
        } else if (currentMatrix === 'sdp-F') {
            label = `\\(F_{${i}}\\)`;
        } else if (currentMatrix === 'sdp2-A_a') {
            label = `\\(A_{${i}}\\)`;
        } else if (currentMatrix === 'sdp2-A_da') {
            label = `\\(A_{${i}}(\\delta)\\)`;
        } else {
            label = `Matrix ${i}`;
        }
        btn.textContent = label;
        btn.onclick = function () {
            openSDPMatrixEditor(i);
        };
        container.appendChild(btn);
    }
    MathJax.typesetPromise();
}

function openSDPModal(tab, matrix) {
    currentMatrix = `${tab}-${matrix}`;
    const matrixInput = document.getElementById(currentMatrix);
    let matrixValues;

    document.getElementById('modal-file').value = '';

    // Create dynamic buttons
    updateSDPButtons();

    // Set default dimensions based on the matrix
    if (matrix === 'F_d') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0, 0]];
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = matrixValues[0].length;
        document.getElementById('matrixColumns').disabled = false;
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit \\(F_j(\\delta)\\) (${tab.toUpperCase()})`;
        document.querySelector('#SDPModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        MathJax.typesetPromise();
        $('#SDPModal').modal('show');
    } else if (matrix === 'F') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0, 0]];
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = matrixValues[0].length;
        document.getElementById('matrixColumns').disabled = false;
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit \\(F_j\\) (${tab.toUpperCase()})`;
        document.querySelector('#SDPModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        MathJax.typesetPromise();
        $('#SDPModal').modal('show');
    } else if (matrix === 'A_da') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0, 0]];
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = matrixValues[0].length;
        document.getElementById('matrixColumns').disabled = false;
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit \\(A_j(\\delta)\\) (${tab.toUpperCase()})`;
        document.querySelector('#SDPModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        MathJax.typesetPromise();
        $('#SDPModal').modal('show');
    } else if (matrix === 'A_a') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0, 0]];
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = matrixValues[0].length;
        document.getElementById('matrixColumns').disabled = false;
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit \\(A_j\\) (${tab.toUpperCase()})`;
        document.querySelector('#SDPModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        MathJax.typesetPromise();
        $('#SDPModal').modal('show');
    } else if (matrix === 'b_da') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0, 0]];
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = 1;
        document.getElementById('matrixColumns').disabled = true;
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit \\(b_j(\\delta)\\) (${tab.toUpperCase()})`;
        document.querySelector('#SDPModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        MathJax.typesetPromise();
        $('#SDPModal').modal('show');
    } else if (matrix === 'b_a') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0, 0]];
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = 1;
        document.getElementById('matrixColumns').disabled = true;
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit \\(b_j\\) (${tab.toUpperCase()})`;
        document.querySelector('#SDPModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        MathJax.typesetPromise();
        $('#SDPModal').modal('show');
    }
}

function openSDPMatrixEditor(idx) {

    if (currentMatrix.endsWith('F_d') || currentMatrix.endsWith('A_da')) {
        if (!window.sdpMatrixCollection1) {
            window.sdpMatrixCollection1 = {};
        }

        window.sdpCurrentMatrixIndex = idx;

        let values = window.sdpMatrixCollection1[idx] ||
            Array.from({length: parseInt(document.getElementById('SDPColumns').value)},
                () => Array(parseInt(document.getElementById('SDPColumns').value)).fill(0));
        document.getElementById('matrixRows').value = values.length;
        document.getElementById('matrixColumns').value = values[0].length;
        document.getElementById('matrixColumns').disabled = false;
        updateMatrixGrid(values);
        document.getElementById('matrixModalTitle').textContent = `Edit Matrix ${idx}`;
        $('#matrixModal').modal('show');
    } else if (currentMatrix.endsWith('F') || currentMatrix.endsWith('A_a')) {
        if (!window.sdpMatrixCollection2) {
            window.sdpMatrixCollection2 = {};
        }

        window.sdpCurrentMatrixIndex = idx;

        let values = window.sdpMatrixCollection2[idx] ||
            Array.from({length: parseInt(document.getElementById('SDPColumns').value)},
                () => Array(parseInt(document.getElementById('SDPColumns').value)).fill(0));
        document.getElementById('matrixRows').value = values.length;
        document.getElementById('matrixColumns').value = values[0].length;
        document.getElementById('matrixColumns').disabled = false;
        updateMatrixGrid(values);
        document.getElementById('matrixModalTitle').textContent = `Edit Matrix ${idx}`;
        $('#matrixModal').modal('show');
    } else if (currentMatrix.endsWith('b_da')) {
        if (!window.sdpMatrixCollection3) {
            window.sdpMatrixCollection3 = {};
        }
        window.sdpCurrentMatrixIndex = idx;

        let values = window.sdpMatrixCollection3[idx] ||
            Array.from({length: parseInt(document.getElementById('SDPColumns').value)},
                () => Array(parseInt(document.getElementById('SDPColumns').value)).fill(0));
        document.getElementById('matrixRows').value = values.length;
        document.getElementById('matrixColumns').value = values[0].length;
        document.getElementById('matrixColumns').disabled = false;
        updateMatrixGrid(values);
        document.getElementById('matrixModalTitle').textContent = `Edit Matrix ${idx}`;
        $('#matrixModal').modal('show');
    } else if (currentMatrix.endsWith('b_a')) {
        if (!window.sdpMatrixCollection4) {
            window.sdpMatrixCollection4 = {};
        }

        window.sdpCurrentMatrixIndex = idx;

        let values = window.sdpMatrixCollection4[idx] ||
            Array.from({length: parseInt(document.getElementById('SDPColumns').value)},
                () => Array(parseInt(document.getElementById('SDPColumns').value)).fill(0));
        document.getElementById('matrixRows').value = values.length;
        document.getElementById('matrixColumns').value = values[0].length;
        document.getElementById('matrixColumns').disabled = false;
        updateMatrixGrid(values);
        document.getElementById('matrixModalTitle').textContent = `Edit Matrix ${idx}`;
        $('#matrixModal').modal('show');
    }
}

function openMatrixModal(tab, matrix) {
    currentMatrix = `${tab}-${matrix}`;
    const matrixInput = document.getElementById(currentMatrix);
    let matrixValues;

    document.getElementById('modal-file').value = '';
    // Set default dimensions based on the matrix
    if (matrix === 'A_d') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0, 0]]; // 1 row, 2 columns
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = matrixValues[0].length;
        document.getElementById('matrixColumns').disabled = false; // Disable column input to prevent changes
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit A(\\(\\delta)\\) (${tab.toUpperCase()})`;
        document.querySelector('#matrixModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        MathJax.typesetPromise();
        $('#matrixModal').modal('show');
    } else if (matrix === 'b_d') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0]]; // 1 row, 1 column
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = matrixValues[0].length;
        document.getElementById('matrixColumns').disabled = true; // Disable column input to prevent changes
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit b(\\(\\delta)\\) (${tab.toUpperCase()})`;
        document.querySelector('#matrixModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        MathJax.typesetPromise();
        $('#matrixModal').modal('show');
    } else if (matrix === 'c') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0], [0]]; // 1 row, 1 column
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = matrixValues[0].length;
        document.getElementById('matrixColumns').disabled = true; // Disable row input to prevent changes
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit c (${tab.toUpperCase()})`;
        document.querySelector('#matrixModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        MathJax.typesetPromise();
        $('#matrixModal').modal('show');
    } else if (matrix === 'Q') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0, 0], [0, 0]]; // 2x2 by default
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = matrixValues[0].length;
        document.getElementById('matrixColumns').disabled = false; // Disable column input to prevent changes
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit Q (${tab.toUpperCase()})`;
        document.querySelector('#matrixModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        MathJax.typesetPromise();
        $('#matrixModal').modal('show');
    } else if (matrix === 'A') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0, 0]]; // 2x2 by default
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = matrixValues[0].length;
        document.getElementById('matrixColumns').disabled = false; // Disable column input to prevent changes
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit G (${tab.toUpperCase()})`;
        document.querySelector('#matrixModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        $('#matrixModal').modal('show');
    } else if (matrix === 'b') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0]]; // 1 row, 1 column
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = matrixValues[0].length;
        document.getElementById('matrixColumns').disabled = true; // Disable row input to prevent changes
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit h (${tab.toUpperCase()})`;
        document.querySelector('#matrixModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        $('#matrixModal').modal('show');
    } else if (matrix === 'C') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0, 0], [0, 0]]; // 2x2 by default
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = matrixValues[0].length;
        document.getElementById('matrixColumns').disabled = false; // Disable column input to prevent changes
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit C (${tab.toUpperCase()})`;
        document.querySelector('#matrixModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        MathJax.typesetPromise();
        $('#matrixModal').modal('show');
    } else if (matrix === 'Theta-bar') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0, 0], [0, 0]]; // 2x2 by default
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = matrixValues[0].length;
        document.getElementById('matrixColumns').disabled = false; // Disable column input to prevent changes
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit \\(\\bar{X}\\) (${tab.toUpperCase()})`;
        document.querySelector('#matrixModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        MathJax.typesetPromise();
        $('#matrixModal').modal('show');
    } else if (matrix === 'theta-bar') {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0], [0]]; // 1 row, 1 column
        document.getElementById('matrixRows').value = matrixValues.length;
        document.getElementById('matrixColumns').value = matrixValues[0].length;
        document.getElementById('matrixColumns').disabled = true; // Disable row input to prevent changes
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `Edit \\(\\bar{x}\\) (${tab.toUpperCase()})`;
        document.querySelector('#matrixModal .modal-body p').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        MathJax.typesetPromise();
        $('#matrixModal').modal('show');
    }
}

function getDynamicTip(matrix) {
    if (matrix === 'A_d') {
        return "<i>Use delta[0] for \\(\\delta_1\\), delta[1] for \\(\\delta_2\\), etc.</i> The matrix can accept any SymPy expressions, e.g., with +, -, *, /, **.  You may manually create the matrix of constraints or upload the constraints using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'b_d') {
        return "<i>Use delta[0] for \\(\\delta_1\\), delta[1] for \\(\\delta_2\\), etc.</i> The vector can accept any SymPy expressions, e.g., with +, -, *, /, **. You may manually create the matrix of constraints or upload the constraints using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'Q') {
        return "<i>Ensure Q is a symmetric and positive semi-definite matrix.</i> You may manually create the matrix or upload using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'c') {
        return "You may manually create the vector or upload using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'A') {
        return "You may manually create the matrix or upload using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'b') {
        return "You may manually create the vector or upload using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'F_d') {
        return "<i>Ensure \\(F_j(\\delta)\\) is a symmetric and positive semi-definite matrix. Use delta[0] for \\(\\delta_1\\), delta[1] for \\(\\delta_2\\), etc.</i> The matrices can accept any SymPy expressions, e.g., with +, -, *, /, **. You may manually create the matrices of constraints or upload the constraints using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'F') {
        return "<i>Ensure \\(F_j\\) is a symmetric and positive semi-definite matrix.</i> You may manually create the matrices of constraints or upload the constraints using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'A_da') {
        return "<i>Ensure \\(A_j(\\delta)\\) is a symmetric and positive semi-definite matrix. Use delta[0] for \\(\\delta_1\\), delta[1] for \\(\\delta_2\\), etc.</i> The matrices can accept any SymPy expressions, e.g., with +, -, *, /, **. You may manually create the matrices of constraints or upload the constraints using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'A_a') {
        return "<i>Ensure \\(A_j\\) is a symmetric and positive semi-definite matrix.</i> You may manually create the matrices of constraints or upload the constraints using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'C') {
        return "<i>Ensure C is a symmetric and positive semi-definite matrix.</i> You may manually create the matrix or upload using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'Theta-bar') {
        return "You may manually create the matrices or upload using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    }else if (matrix === 'theta-bar') {
        return "You may manually create the vector or upload using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'b_da') {
        return "<i>Use delta[0] for \\(\\delta_1\\), delta[1] for \\(\\delta_2\\), etc.</i> The vector can accept any SymPy expressions, e.g., with +, -, *, /, **. You may manually create the matrix of constraints or upload the constraints using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'b_a') {
        return "You may manually create the vectors or upload using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    }
    return "Provide valid input for the selected matrix.";
}

function updateMatrixGrid(values = null) {
    const rows = parseInt(document.getElementById('matrixRows').value);
    const columns = parseInt(document.getElementById('matrixColumns').value);
    const grid = document.getElementById('matrixGrid');

    grid.innerHTML = ''; // Clear the grid

    for (let i = 0; i < rows; i++) {
        const rowDiv = document.createElement('div'); // Create a div for each row
        rowDiv.className = 'matrix-row'; // Add a class for styling

        for (let j = 0; j < columns; j++) {
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'form-control d-inline-block';
            input.style.width = '120px';
            input.value = values && values[i] && values[i][j] !== undefined ? values[i][j] : 0;
            input.dataset.row = i;
            input.dataset.column = j;
            rowDiv.appendChild(input); // Append input to the row div
        }

        grid.appendChild(rowDiv); // Append the row div to the grid
    }
}

function saveCollection() {
    // This function saves the current matrix collection for SDP problems.
    // It supports uploading matrix data from a file (JSON, CSV, TXT) or saving manually entered data.

    // Get the file input element for uploading matrix data
    const fileInput = document.getElementById('SDP-file');

    // Check if a file is uploaded
    if (fileInput && fileInput.files.length > 0) {
        const file = fileInput.files[0]; // Get the uploaded file
        const reader = new FileReader(); // Create a FileReader to read the file

        // Define what happens when the file is read
        reader.onload = function (event) {
            try {
                // Determine the file extension to handle different formats
                const fileExtension = file.name.split('.').pop().toLowerCase();
                let fileValues;

                // Parse JSON files directly
                if (fileExtension === 'json') {
                    fileValues = JSON.parse(event.target.result);
                    // Parse CSV or TXT files by splitting into rows and columns
                } else if (fileExtension === 'csv' || fileExtension === 'txt') {
                    const text = event.target.result;
                    const rows = text.trim().split('\n');
                    fileValues = rows.map(row => row.split(',').map(cell => cell.trim()));
                } else {
                    // Unsupported file type
                    throw new Error('Unsupported file type');
                }

                if (currentMatrix.endsWith('F_d') || currentMatrix.endsWith('A_da')) {
                    // Overwrite the matrix collection for the selected tab with the uploaded data
                    window.sdpMatrixCollection1 = fileValues;
                } else if (currentMatrix.endsWith('F') || currentMatrix.endsWith('A_a')) {
                    // Overwrite the matrix collection for the selected tab with the uploaded data
                    window.sdpMatrixCollection2 = fileValues;
                } else if (currentMatrix.endsWith('b_da')) {
                    // Overwrite the matrix collection for the selected tab with the uploaded data
                    window.sdpMatrixCollection3 = fileValues;
                } else if (currentMatrix.endsWith('b_a')) {
                    // Overwrite the matrix collection for the selected tab with the uploaded data
                    window.sdpMatrixCollection4 = fileValues;
                }


            } catch (error) {
                // Show an error if the file format is invalid
                alert('Invalid file format. Please upload a valid JSON, CSV, or TXT file.');
                return;
            }
            // Update the hidden input with the new matrix collection and close the modal


            if (currentMatrix.endsWith('F_d') || currentMatrix.endsWith('A_da')) {
                // Overwrite the matrix collection for the selected tab with the uploaded data
                document.getElementById(currentMatrix).value = JSON.stringify(window.sdpMatrixCollection1);
            } else if (currentMatrix.endsWith('F') || currentMatrix.endsWith('A_a')) {
                // Overwrite the matrix collection for the selected tab with the uploaded data
                document.getElementById(currentMatrix).value = JSON.stringify(window.sdpMatrixCollection2);
            } else if (currentMatrix.endsWith('b_da')) {
                // Overwrite the matrix collection for the selected tab with the uploaded data
                document.getElementById(currentMatrix).value = JSON.stringify(window.sdpMatrixCollection3);
            } else if (currentMatrix.endsWith('b_a')) {
                // Overwrite the matrix collection for the selected tab with the uploaded data
                document.getElementById(currentMatrix).value = JSON.stringify(window.sdpMatrixCollection4);
            }
            $('#SDPModal').modal('hide');
        };
        // Start reading the file as text
        reader.readAsText(file);
    } else {
        // If no file is uploaded, just save the current matrix collection to the hidden input and close the modal
        if (currentMatrix.endsWith('F_d') || currentMatrix.endsWith('A_da')) {
            // Overwrite the matrix collection for the selected tab with the uploaded data
            document.getElementById(currentMatrix).value = JSON.stringify(window.sdpMatrixCollection1);
        } else if (currentMatrix.endsWith('F') || currentMatrix.endsWith('A_a')) {
            // Overwrite the matrix collection for the selected tab with the uploaded data
            document.getElementById(currentMatrix).value = JSON.stringify(window.sdpMatrixCollection2);
        } else if (currentMatrix.endsWith('b_da')) {
            // Overwrite the matrix collection for the selected tab with the uploaded data
            document.getElementById(currentMatrix).value = JSON.stringify(window.sdpMatrixCollection3);
        } else if (currentMatrix.endsWith('b_a')) {
            // Overwrite the matrix collection for the selected tab with the uploaded data
            document.getElementById(currentMatrix).value = JSON.stringify(window.sdpMatrixCollection4);
        }
        $('#SDPModal').modal('hide');
    }
}

function saveMatrix() {
    const rows = parseInt(document.getElementById('matrixRows').value);
    const columns = parseInt(document.getElementById('matrixColumns').value);
    const grid = document.getElementById('matrixGrid');
    const fileInput = document.getElementById('modal-file');
    const values = Array.from({length: rows}, () => Array(columns).fill(0));

    // Check if a file is uploaded
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const reader = new FileReader();

        reader.onload = function (event) {
            try {
                const fileExtension = file.name.split('.').pop().toLowerCase();
                let fileValues;

                if (fileExtension === 'json') {
                    fileValues = JSON.parse(event.target.result);
                } else if (fileExtension === 'csv' || fileExtension === 'txt') {
                    const text = event.target.result;
                    const rows = text.trim().split('\n');
                    fileValues = rows.map(row => row.split(','));
                } else {
                    throw new Error('Unsupported file type');
                }

                // If editing from SDP modal, store in collection
                if (window.sdpCurrentMatrixIndex !== null) {

                    // If no file is uploaded, just save the current matrix collection to the hidden input and close the modal
                    if (currentMatrix.endsWith('F_d') || currentMatrix.endsWith('A_da')) {
                        window.sdpMatrixCollection1[window.sdpCurrentMatrixIndex] = fileValues;
                    } else if (currentMatrix.endsWith('F') || currentMatrix.endsWith('A_a')) {
                        window.sdpMatrixCollection2[window.sdpCurrentMatrixIndex] = fileValues;
                    } else if (currentMatrix.endsWith('b_da')) {
                        window.sdpMatrixCollection3[window.sdpCurrentMatrixIndex] = fileValues;
                    } else if (currentMatrix.endsWith('b_a')) {
                        window.sdpMatrixCollection4[window.sdpCurrentMatrixIndex] = fileValues;
                    }

                    window.sdpCurrentMatrixIndex = null;
                    $('#matrixModal').modal('hide');
                    document.getElementById('SDP-file').value = '';
                    return;
                }

                document.getElementById(currentMatrix).value = JSON.stringify(fileValues);
                $('#matrixModal').modal('hide');
                document.getElementById('SDP-file').value = '';
            } catch (error) {
                alert('Invalid file format. Please upload a valid JSON, CSV, or TXT file.');
            }
        };

        reader.readAsText(file); // Read the file content
    } else {
        // Save manually entered values
        Array.from(grid.querySelectorAll('input')).forEach(input => {
            const row = parseInt(input.dataset.row);
            const column = parseInt(input.dataset.column);
            values[row][column] = input.value; // Save text value
        });

        // If editing from SDP modal, store in collection
        if (window.sdpCurrentMatrixIndex !== null) {


            // If no file is uploaded, just save the current matrix collection to the hidden input and close the modal
            if (currentMatrix.endsWith('F_d') || currentMatrix.endsWith('A_da')) {
                window.sdpMatrixCollection1[window.sdpCurrentMatrixIndex] = values;
            } else if (currentMatrix.endsWith('F') || currentMatrix.endsWith('A_a')) {
                window.sdpMatrixCollection2[window.sdpCurrentMatrixIndex] = values;
            } else if (currentMatrix.endsWith('b_da')) {
                window.sdpMatrixCollection3[window.sdpCurrentMatrixIndex] = values;
            } else if (currentMatrix.endsWith('b_a')) {
                window.sdpMatrixCollection4[window.sdpCurrentMatrixIndex] = values;
            }

            window.sdpCurrentMatrixIndex = null;
            $('#matrixModal').modal('hide');
            document.getElementById('SDP-file').value = '';
            return;
        }
        document.getElementById(currentMatrix).value = JSON.stringify(values);
        $('#matrixModal').modal('hide');
        document.getElementById('SDP-file').value = '';
    }
}

function solve() {
    const activeTab = document.querySelector('.nav-link.active').id;
    let formId;
    if (activeTab === 'lp-tab') {
        formId = 'lp-form';
    } else if (activeTab === 'qp-tab') {
        formId = 'qp-form';
    } else if (activeTab === 'sdp-tab') {
        formId = 'sdp-form';
    } else if (activeTab === 'sdp2-tab') {
        formId = 'sdp2-form';
    }

    const activeForm = document.getElementById(formId);
    const formData = new FormData(activeForm);

    // Add solver form data
    const solverForm = document.getElementById('solver-form');
    const solverData = new FormData(solverForm);
    for (const [key, value] of solverData.entries()) {
        formData.append(key, value);
    }

    const fileInput = document.getElementById('file');
    if (fileInput && fileInput.files.length > 0) {
        formData.append('file', fileInput.files[0]);
    }

    document.getElementById('result-box').innerHTML = '<p>Solve button pressed, processing results...</p>';

    fetch(activeForm.action, {
        method: activeForm.method,
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            const resultBox = document.getElementById('result-box');
            resultBox.innerHTML = `
    <div class="result-container">
    <ul class="nav nav-tabs" id="resultTab" role="tablist">
        <li class="nav-item">
            <a class="nav-link active" id="result-table-tab" data-toggle="tab" href="#result-table" role="tab" aria-controls="result-table" aria-selected="true">Results Table</a>
        </li>
        <li class="nav-item">
            <a class="nav-link" id="result-graph-tab" data-toggle="tab" href="#result-graph" role="tab" aria-controls="result-graph" aria-selected="false">ρ Graph</a>
        </li>
        <li class="nav-item">
            <a class="nav-link" id="result-graph2-tab" data-toggle="tab" href="#result-graph2" role="tab" aria-controls="result-graph2" aria-selected="false">𝜏 Graph</a>
        </li>
        <li class="nav-item">
            <a class="nav-link" id="result-formdata-tab" data-toggle="tab" href="#result-formdata" role="tab" aria-controls="result-formdata" aria-selected="false">Form Data</a>
        </li>
    </ul>
    <div class="tab-content" id="resultTabContent">
        <div class="tab-pane fade show active" id="result-table" role="tabpanel" aria-labelledby="result-table-tab">
            <h3><br> Optimization Results</h3>
            <div class="result-table-container">
            ${generateResultTable(data)}
            </div>
        </div>
        <div class="tab-pane fade" id="result-graph" role="tabpanel" aria-labelledby="result-graph-tab">
            <h3><br> ρ Graph</h3>
            <div class="form-group">
            <label for="tau-select">Select 𝜏:</label>
            <select id="tau-select" class="form-control" style="width: 200px; display: inline-block;"></select>
            </div>
            <canvas id="resultGraph" width="400" height="300"></canvas>
        </div>
        <div class="tab-pane fade" id="result-graph2" role="tabpanel" aria-labelledby="result-graph2-tab">
            <h3><br> 𝜏 Graph</h3>
            <div class="form-group">
            <label for="rho-select">Select ρ:</label>
            <select id="rho-select" class="form-control" style="width: 200px; display: inline-block;"></select>
            </div>
            <canvas id="resultGraph2" width="400" height="300"></canvas>
        </div>
        <div class="tab-pane fade" id="result-formdata" role="tabpanel" aria-labelledby="result-formdata-tab">
            <h3><br> Form Data</h3>
            <pre style="white-space: pre-wrap; word-break: break-all;">${JSON.stringify(data.form_data, null, 2)}</pre>
        </div>
    </div>
</div>
`;

            // Prepare risk lower and upper arrays
            const riskLower = Array.isArray(data.risk) ? data.risk.map(r => Array.isArray(r) ? r[0] : r) : [data.risk];
            const riskUpper = Array.isArray(data.risk) ? data.risk.map(r => Array.isArray(r) ? r[1] : r) : [data.risk];

            // Assume data.rho_ and data.tau_ are arrays of equal length, and each (rho, tau) pair is unique
            const uniqueTau = Array.from(new Set(Array.isArray(data.tau_) ? data.tau_ : [data.tau_]));
            const tauSelect = document.getElementById('tau-select');
            tauSelect.innerHTML = uniqueTau.map(tau => `<option value="${tau}">${tau}</option>`).join('');

// Helper to filter data by selected tau
            function getFilteredByTau(selectedTau) {
                const indices = [];
                (Array.isArray(data.tau_) ? data.tau_ : [data.tau_]).forEach((tau, i) => {
                    if (String(tau) === String(selectedTau)) indices.push(i);
                });
                return {
                    rho: indices.map(i => Array.isArray(data.rho_) ? data.rho_[i] : data.rho_),
                    riskLower: indices.map(i => Array.isArray(riskLower) ? riskLower[i] : riskLower),
                    riskUpper: indices.map(i => Array.isArray(riskUpper) ? riskUpper[i] : riskUpper),
                    cost: indices.map(i => Array.isArray(data.optimal_cost) ? data.optimal_cost[i] : data.optimal_cost)
                };
            }

// Draw chart for selected rho
            let chart;

            function drawRhoChart(selectedTau) {
                const filtered = getFilteredByTau(selectedTau);
                const ctx = document.getElementById('resultGraph').getContext('2d');
                if (chart) chart.destroy();
                chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: filtered.rho,
                        datasets: [
                            {
                                label: 'Risk Lower Bound (εₗ)',
                                data: filtered.riskLower,
                                borderColor: 'rgba(75, 192, 192, 1)',
                                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                                yAxisID: 'y-risk',
                                fill: false,
                                tension: 0.1
                            },
                            {
                                label: 'Risk Upper Bound (εᵤ)',
                                data: filtered.riskUpper,
                                borderColor: 'rgba(54, 162, 235, 1)',
                                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                                yAxisID: 'y-risk',
                                fill: '-1',
                                tension: 0.1
                            },
                            {
                                label: 'Cost',
                                data: filtered.cost,
                                borderColor: 'rgba(36, 32, 34, 0.8)',
                                yAxisID: 'y-cost',
                                fill: false,
                                tension: 0.1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        interaction: {mode: 'index', intersect: false},
                        scales: {
                            x: {
                                title: {display: true, text: 'Relaxation Term (ρ)'},
                                type: 'linear',
                                position: 'bottom'
                            },
                            'y-risk': {
                                type: 'linear',
                                position: 'left',
                                title: {display: true, text: 'Risk (ε)'}
                            },
                            'y-cost': {
                                type: 'linear',
                                position: 'right',
                                title: {display: true, text: 'Cost'},
                                grid: {drawOnChartArea: false}
                            }
                        }
                    }
                });
            }

// Initial draw
            drawRhoChart(uniqueTau[0]);
            tauSelect.addEventListener('change', e => drawRhoChart(e.target.value));

            // Assume data.rho_ and data.tau_ are arrays of equal length, and each (rho, tau) pair is unique
            const uniqueRho = Array.from(new Set(Array.isArray(data.rho_) ? data.rho_ : [data.rho_]));
            const rhoSelect = document.getElementById('rho-select');
            rhoSelect.innerHTML = uniqueRho.map(rho => `<option value="${rho}">${rho}</option>`).join('');

// Helper to filter data by selected rho
            function getFilteredByRho(selectedRho) {
                const indices = [];
                (Array.isArray(data.rho_) ? data.rho_ : [data.rho_]).forEach((rho, i) => {
                    if (String(rho) === String(selectedRho)) indices.push(i);
                });
                return {
                    tau: indices.map(i => Array.isArray(data.tau_) ? data.tau_[i] : data.tau_),
                    riskLower: indices.map(i => Array.isArray(riskLower) ? riskLower[i] : riskLower),
                    riskUpper: indices.map(i => Array.isArray(riskUpper) ? riskUpper[i] : riskUpper),
                    cost: indices.map(i => Array.isArray(data.optimal_cost) ? data.optimal_cost[i] : data.optimal_cost)
                };
            }

// Draw chart for selected rho
            let chart2;

            function drawTauChart(selectedRho) {
                const filtered = getFilteredByRho(selectedRho);
                const ctx2 = document.getElementById('resultGraph2').getContext('2d');
                if (chart2) chart2.destroy();
                chart2 = new Chart(ctx2, {
                    type: 'line',
                    data: {
                        labels: filtered.tau,
                        datasets: [
                            {
                                label: 'Risk Lower Bound (εₗ)',
                                data: filtered.riskLower,
                                borderColor: 'rgba(75, 192, 192, 1)',
                                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                                yAxisID: 'y-risk',
                                fill: false,
                                tension: 0.1
                            },
                            {
                                label: 'Risk Upper Bound (εᵤ)',
                                data: filtered.riskUpper,
                                borderColor: 'rgba(54, 162, 235, 1)',
                                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                                yAxisID: 'y-risk',
                                fill: '-1',
                                tension: 0.1
                            },
                            {
                                label: 'Cost',
                                data: filtered.cost,
                                borderColor: 'rgba(36, 32, 34, 0.8)',
                                yAxisID: 'y-cost',
                                fill: false,
                                tension: 0.1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        interaction: {mode: 'index', intersect: false},
                        scales: {
                            x: {
                                title: {display: true, text: 'Regularization Term (τ)'},
                                type: 'linear',
                                position: 'bottom'
                            },
                            'y-risk': {
                                type: 'linear',
                                position: 'left',
                                title: {display: true, text: 'Risk (ε)'}
                            },
                            'y-cost': {
                                type: 'linear',
                                position: 'right',
                                title: {display: true, text: 'Cost'},
                                grid: {drawOnChartArea: false}
                            }
                        }
                    }
                });
            }

// Initial draw
            drawTauChart(uniqueRho[0]);
            rhoSelect.addEventListener('change', e => drawTauChart(e.target.value));
            MathJax.typesetPromise();
        })
        .catch(error => console.error('Error:', error));
}

function generateResultTable(data) {
    // Determine the number of value columns
    const count = data.count || 1;
    const valueHeaders = Array.from({length: count}, (_, i) => `<th>Value${i + 1}</th>`).join('');

    // Prepare rows with values as arrays (or wrap single values)
    const rows = [
        {label: 'Optimal Cost', values: Array.isArray(data.optimal_cost) ? data.optimal_cost : [data.optimal_cost]},
        {label: 'Optimal &theta;', values: Array.isArray(data.optimal_x) ? data.optimal_x : [data.optimal_x]},
        {label: 'Optimal &zeta;', values: Array.isArray(data.optimal_s) ? data.optimal_s : [data.optimal_s]},
        {label: 'Relaxation Parameters &rho;', values: Array.isArray(data.rho_) ? data.rho_ : [data.rho_]},
        {label: 'Regularization Parameters &tau;', values: Array.isArray(data.tau_) ? data.tau_ : [data.tau_]},
        {
            label: 'Confidence \\(1-\\frac{\\beta}{n_{\\tau} n_{\\rho}}\\)',
            values: Array.isArray(data.conf) ? data.conf : [data.conf]
        },
        {label: 'Risk Bounds &epsilon;', values: Array.isArray(data.risk) ? data.risk : [data.risk]},
        {
            label: 'Degeneracy Detected?',
            values: Array.isArray(data.degeneracy) ? data.degeneracy : [data.degeneracy]
        },
        {label: 'Support Set Size', values: Array.isArray(data.active_con) ? data.active_con : [data.active_con]},
        {
            label: 'Number of data samples',
            values: Array.isArray(data.num_deltas) ? data.num_deltas : [data.num_deltas]
        },
        {label: 'Total Constraints', values: Array.isArray(data.tot_con) ? data.tot_con : [data.tot_con]},
        {label: '<i>Error</i>', values: Array.isArray(data.errorcode) ? data.errorcode : [data.errorcode]}//,
        // { label: 'Form Data', values: [JSON.stringify(data.form_data)] }
    ];

    let table = `<table class="result-table">
            <tr>
                <th>Parameter</th>
                ${valueHeaders}
            </tr>`;
    rows.forEach(row => {
        if (row.values.length === 1 && count > 1) {
            row.values = Array(count).fill(row.values[0]);
        }
        table += `
            <tr>
                <td>${row.label}</td>
                ${row.values.map((val, i) => {
            // Risk column: show as [a, b]
            if (row.label.includes('Risk Bounds')) {
                if (Array.isArray(val) && val.length === 2) {
                    return `<td><strong>[${val[0]}, ${val[1]}]</strong></td>`;
                }
                return `<td>${val}</td>`;
            }
            if (row.label.includes('Optimal Cost')) {
                return `<td><strong>${val}</strong></td>`;
            }
            // Error column: all inside <i>
            if (row.label.includes('Error')) {
                if (Array.isArray(val)) {
                    return `<td><i>${val.join('<br>')}</i></td>`;
                }
                return `<td><i>${val}</i></td>`;
            }
            if (row.label.includes('Degeneracy Detected?')) {
                if (Array.isArray(val) && val.includes("true")) {
                    return `<td><i>${val}</i></td>`; // TODO: make true appear in a different color!
                }
                return `<td>${val}</td>`;
            }

            // All other columns: newline separated if multiple
            if (Array.isArray(val)) {
                return `<td><strong>${val.join('<br>')}<strong></td>`;
            }
            return `<td>${val !== undefined ? val : ''}</td>`;
        }).join('')}
            </tr>`;
    });
    table += `</table>`;
    return table;
}

function updateLatexText(tab) {
    const option = document.getElementById(`${tab}-options`).value;
    let latexText1 = '';
    let latexText2 = '';
    let latexText3 = '';
    let latexText4 = '';

    // Show/Hide relevant input fields
    document.getElementById(`${tab}-tau-group`).style.display = 'none';
    document.getElementById(`${tab}-theta-bar-group`).style.display = 'none';
    document.getElementById(`${tab}-rho-group`).style.display = 'none';
    document.getElementById(`${tab}-p-group`).style.display = 'none';

    if (tab === 'lp') {
        if (option === 'robust') {
            latexText1 = '\\displaystyle\\min_{x} c^\\top x';
            latexText2 = 'A(\\delta_i)x+b(\\delta_i) \\leq 0, \\quad i = 1, \\ldots, N';
            latexText4 = 'Ax+b \\leq 0';
        } else if (option === 'robust-regularization') {
            latexText1 = '\\displaystyle\\min_{x} c^\\top x + \\tau\\Vert x - \\bar{x}\\Vert_{p}';
            latexText2 = 'A(\\delta_i)x+b(\\delta_i) \\leq 0, \\quad i = 1, \\ldots, N';
            latexText4 = 'Ax+b \\leq 0';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        } else if (option === 'robust-relaxation') {
            latexText1 = '\\displaystyle\\min_{x,\\zeta_i} c^\\top x + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText2 = 'A(\\delta_i)x+b(\\delta_i) \\leq \\zeta_i, \\quad i = 1, \\ldots, N';
            latexText3 = '\\zeta_i \\geq 0,';
            latexText4 = 'Ax+b \\leq 0';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
        } else if (option === 'robust-regularization-relaxation') {
            latexText1 = '\\displaystyle\\min_{x,\\zeta_i} c^\\top x + \\tau\\Vert x - \\bar{x}\\Vert_{p} + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText2 = 'A(\\delta_i)x+b(\\delta_i) \\leq \\zeta_i, \\quad i = 1, \\ldots, N';
            latexText3 = '\\zeta_i \\geq 0,';
            latexText4 = 'Ax+b \\leq 0';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        }
    } else if (tab === 'qp') {
        if (option === 'robust') {
            latexText1 = '\\displaystyle\\min_{x} c^\\top x + \\frac{1}{2}x^\\top Qx';
            latexText2 = 'A(\\delta_i)x+b(\\delta_i) \\leq 0, \\quad i = 1, \\ldots, N';
            latexText3 = '';
            latexText4 = 'Ax+b \\leq 0';
        } else if (option === 'robust-regularization') {
            latexText1 = '\\displaystyle\\min_{x} c^\\top x + \\frac{1}{2}x^\\top Qx + \\tau\\Vert x - \\bar{x}\\Vert_{p}';
            latexText2 = 'A(\\delta_i)x+b(\\delta_i) \\leq 0, \\quad i = 1, \\ldots, N';
            latexText3 = '';
            latexText4 = 'Ax+b \\leq 0';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        } else if (option === 'robust-relaxation') {
            latexText1 = '\\displaystyle\\min_{x,\\zeta_i} c^\\top x + \\frac{1}{2}x^\\top Qx + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText2 = 'A(\\delta_i)x+b(\\delta_i) \\leq \\zeta_i, \\quad i = 1, \\ldots, N';
            latexText3 = '\\zeta_i \\geq 0,';
            latexText4 = 'Ax+b \\leq 0';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
        } else if (option === 'robust-regularization-relaxation') {
            latexText1 = '\\displaystyle\\min_{x,\\zeta_i} c^\\top x + \\frac{1}{2}x^\\top Qx + \\tau\\Vert x - \\bar{x}\\Vert_{p} + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText2 = 'A(\\delta_i)x+b(\\delta_i) \\leq \\zeta_i, \\quad i = 1, \\ldots, N';
            latexText3 = '\\zeta_i \\geq 0';
            latexText4 = 'Ax+b \\leq 0';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        }
    } else if (tab === 'sdp') {
        if (option === 'robust') {
            latexText1 = '\\displaystyle\\min_{x} c^\\top x + \\frac{1}{2}x^\\top Qx';
            latexText2 = 'F_0(\\delta_i) + \\displaystyle\\sum_{j=1}^{n}F_j(\\delta_i)x_j \\leq 0, \\quad i = 1, \\ldots, N';
            latexText3 = '';
            latexText4 = 'F_0 + \\displaystyle\\sum_{j=1}^{n}F_jx_j \\leq 0';
        } else if (option === 'robust-regularization') {
            latexText1 = '\\displaystyle\\min_{x} c^\\top x + \\frac{1}{2}x^\\top Qx + \\tau\\Vert x - \\bar{x}\\Vert_{p}';
            latexText2 = 'F_0(\\delta_i) + \\displaystyle\\sum_{j=1}^{n}F_j(\\delta_i)x_j \\leq 0, \\quad i = 1, \\ldots, N';
            latexText3 = '';
            latexText4 = 'F_0 + \\displaystyle\\sum_{j=1}^{n}F_jx_j \\leq 0';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        } else if (option === 'robust-relaxation') {
            latexText1 = '\\displaystyle\\min_{x,\\zeta_i} c^\\top x + \\frac{1}{2}x^\\top Qx + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText2 = 'F_0(\\delta_i) + \\displaystyle\\sum_{j=1}^{n}F_j(\\delta_i)x_j \\leq \\zeta_i, \\quad i = 1, \\ldots, N';
            latexText3 = '\\zeta_i \\geq 0,';
            latexText4 = 'F_0 + \\displaystyle\\sum_{j=1}^{n}F_jx_j \\leq 0';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
        } else if (option === 'robust-regularization-relaxation') {
            latexText1 = '\\displaystyle\\min_{x,\\zeta_i} c^\\top x + \\frac{1}{2}x^\\top Qx + \\tau\\Vert x - \\bar{x}\\Vert_{p} + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText2 = 'F_0(\\delta_i) + \\displaystyle\\sum_{j=1}^{n}F_j(\\delta_i)x_j \\leq \\zeta_i, \\quad i = 1, \\ldots, N';
            latexText3 = '\\zeta_i \\geq 0,';
            latexText4 = 'F_0 + \\displaystyle\\sum_{j=1}^{n}F_jx_j \\leq 0';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        }
    } else if (tab === 'sdp2') {
        if (option === 'robust') {
            latexText1 = '\\displaystyle\\min_{X} \\textbf{tr}(CX)';
            latexText2 = '\\textbf{tr}(A_j(\\delta_i)X) + b_j(\\delta_i) = 0,~j=1,\\ldots, \\alpha,~i=1,\\ldots,N';
            latexText3 = '';
            latexText4 = '\\textbf{tr}(A_jX) + b_j = 0,~j=1,\\ldots, \\alpha';
        } else if (option === 'robust-regularization') {
            latexText1 = '\\displaystyle\\min_{X} \\textbf{tr}(CX) + \\tau\\Vert X - \\bar{X}\\Vert_{p}';
            latexText2 = '\\textbf{tr}(A_j(\\delta_i)X) + b_j(\\delta_i) = 0,~j=1,\\ldots, \\alpha,~i=1,\\ldots,N';
            latexText3 = '';
            latexText4 = '\\textbf{tr}(A_jX) + b_j = 0,~j=1,\\ldots, \\alpha';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        } else if (option === 'robust-relaxation') {
            latexText1 = '\\displaystyle\\min_{X,\\zeta_i} \\textbf{tr}(CX) + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText2 = '\\textbf{tr}(A_j(\\delta_i)X) + b_j(\\delta_i) = \\zeta_i,~j=1,\\ldots, \\alpha,~i=1,\\ldots,N';
            latexText3 = '\\zeta_i \\geq 0,';
            latexText4 = '\\textbf{tr}(A_jX) + b_j = 0,~j=1,\\ldots, \\alpha';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
        } else if (option === 'robust-regularization-relaxation') {
            latexText1 = '\\displaystyle\\min_{X,\\zeta_i} \\textbf{tr}(CX) + \\tau\\Vert X - \\bar{X}\\Vert_{p} + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText2 = '\\textbf{tr}(A_j(\\delta_i)X) + b_j(\\delta_i) = \\zeta_i,~j=1,\\ldots, \\alpha,~i=1,\\ldots,N';
            latexText3 = '\\zeta_i \\geq 0,';
            latexText4 = '\\textbf{tr}(A_jX) + b_j = 0,~j=1,\\ldots, \\alpha';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        }
    }

    document.getElementById(`${tab}-latex`).innerHTML = `<div class="text-center"><p>\\(${latexText1}\\)</p><p>subject to \\(${latexText3}\\)</p><p>\\(${latexText4}\\)</p><p>\\(${latexText2}\\)</p></div>`;
    MathJax.typeset();
}