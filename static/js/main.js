window.sdpMatrixCollection1 = {};
window.sdpMatrixCollection2 = {};
window.sdpMatrixCollection3 = {};
window.sdpMatrixCollection4 = {};
window.sdpCurrentMatrixIndex = null;
let currentMatrix = '';
let lastResultData = null;

// --- MOSEK License Session Caching ---
const MOSEK_LICENSE_KEY = 'mosek_license_content';
const MOSEK_LICENSE_NAME_KEY = 'mosek_license_filename';

function cacheMosekLicense(file) {
    return new Promise(function(resolve) {
        var reader = new FileReader();
        reader.onload = function(e) {
            try {
                sessionStorage.setItem(MOSEK_LICENSE_KEY, e.target.result);
                sessionStorage.setItem(MOSEK_LICENSE_NAME_KEY, file.name);
                updateMosekCacheIndicator();
            } catch (err) {
                console.warn('Could not cache MOSEK license:', err);
            }
            resolve();
        };
        reader.onerror = function() {
            console.warn('Could not read MOSEK license file for caching');
            resolve();
        };
        reader.readAsText(file);
    });
}

function getCachedMosekLicense() {
    var content = sessionStorage.getItem(MOSEK_LICENSE_KEY);
    var name = sessionStorage.getItem(MOSEK_LICENSE_NAME_KEY) || 'mosek.lic';
    if (!content) return null;
    var blob = new Blob([content], { type: 'application/octet-stream' });
    return new File([blob], name, { type: 'application/octet-stream' });
}

function clearMosekLicenseCache() {
    sessionStorage.removeItem(MOSEK_LICENSE_KEY);
    sessionStorage.removeItem(MOSEK_LICENSE_NAME_KEY);
    updateMosekCacheIndicator();
}

function updateMosekCacheIndicator() {
    var indicator = document.getElementById('mosek-cache-indicator');
    if (!indicator) return;
    var content = sessionStorage.getItem(MOSEK_LICENSE_KEY);
    if (content) {
        var name = sessionStorage.getItem(MOSEK_LICENSE_NAME_KEY) || 'mosek.lic';
        indicator.style.display = 'block';
        indicator.querySelector('.mosek-cache-filename').textContent = name;
    } else {
        indicator.style.display = 'none';
    }
}

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
            label = `\\(E_{${i}}\\)`;
        // NOTE: A_da/A_a branches for SDP2 (standard form) were removed.
        // The SDP modal infrastructure (openSDPModal, openSDPMatrixEditor, etc.) retains
        // A_da/A_a/b_da/b_a handling as it is shared plumbing — now unreachable from the UI
        // but harmless to keep.
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
        document.getElementById('matrixModalTitle').textContent = `Edit \\(E_j\\) (${tab.toUpperCase()})`;
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
        document.getElementById('matrixModalTitle').textContent = `Edit \\(G_j\\) (${tab.toUpperCase()})`;
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
        document.getElementById('matrixModalTitle').textContent = `Edit \\(h_j\\) (${tab.toUpperCase()})`;
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
        return "<i>Ensure \\(E_j\\) is a symmetric and positive semi-definite matrix.</i> You may manually create the matrices of constraints or upload the constraints using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'A_da') {
        return "<i>Ensure \\(A_j(\\delta)\\) is a symmetric and positive semi-definite matrix. Use delta[0] for \\(\\delta_1\\), delta[1] for \\(\\delta_2\\), etc.</i> The matrices can accept any SymPy expressions, e.g., with +, -, *, /, **. You may manually create the matrices of constraints or upload the constraints using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
    } else if (matrix === 'A_a') {
        return "<i>Ensure \\(G_j\\) is a symmetric and positive semi-definite matrix.</i> You may manually create the matrices of constraints or upload the constraints using the 'Upload from File' button, accepted formats are .csv, .txt, .json.";
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

function applyCollectionValues(fileValues) {
    if (currentMatrix.endsWith('F_d') || currentMatrix.endsWith('A_da')) {
        window.sdpMatrixCollection1 = fileValues;
        document.getElementById(currentMatrix).value = JSON.stringify(window.sdpMatrixCollection1);
    } else if (currentMatrix.endsWith('F') || currentMatrix.endsWith('A_a')) {
        window.sdpMatrixCollection2 = fileValues;
        document.getElementById(currentMatrix).value = JSON.stringify(window.sdpMatrixCollection2);
    } else if (currentMatrix.endsWith('b_da')) {
        window.sdpMatrixCollection3 = fileValues;
        document.getElementById(currentMatrix).value = JSON.stringify(window.sdpMatrixCollection3);
    } else if (currentMatrix.endsWith('b_a')) {
        window.sdpMatrixCollection4 = fileValues;
        document.getElementById(currentMatrix).value = JSON.stringify(window.sdpMatrixCollection4);
    }
    $('#SDPModal').modal('hide');
}

function saveCollection() {
    const fileInput = document.getElementById('SDP-file');

    if (fileInput && fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const ext = file.name.split('.').pop().toLowerCase();

        if (BINARY_FORMATS.includes(ext)) {
            parseBinaryFile(file)
                .then(fileValues => applyCollectionValues(fileValues))
                .catch(error => alert('Error parsing file: ' + error.message));
        } else {
            const reader = new FileReader();
            reader.onload = function (event) {
                try {
                    const fileValues = parseTextFile(file, event.target.result);
                    applyCollectionValues(fileValues);
                } catch (error) {
                    alert('Invalid file format: ' + error.message);
                }
            };
            reader.readAsText(file);
        }
    } else {
        applyCollectionValues(undefined);
    }
}

// Text-based formats that can be parsed client-side
const TEXT_FORMATS = ['json', 'csv', 'txt', 'tsv'];
// Binary formats that need server-side parsing via /parse-file
const BINARY_FORMATS = ['npy', 'npz', 'mat', 'xlsx', 'xls', 'parquet'];
const ALL_MATRIX_FORMATS = TEXT_FORMATS.concat(BINARY_FORMATS);

function parseTextFile(file, text) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (ext === 'json') {
        return JSON.parse(text);
    } else if (ext === 'csv' || ext === 'txt') {
        const rows = text.trim().split('\n');
        return rows.map(row => row.split(',').map(cell => cell.trim()));
    } else if (ext === 'tsv') {
        const rows = text.trim().split('\n');
        return rows.map(row => row.split('\t').map(cell => cell.trim()));
    }
    throw new Error('Unsupported text format: ' + ext);
}

function parseBinaryFile(file) {
    // Send to backend for parsing
    const formData = new FormData();
    formData.append('file', file);
    return fetch('/parse-file', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(result => {
            if (result.error) throw new Error(result.error);
            return result.data;
        });
}

function applyMatrixFileValues(fileValues) {
    // If editing from SDP modal, store in collection
    if (window.sdpCurrentMatrixIndex !== null) {
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
        const ext = file.name.split('.').pop().toLowerCase();

        if (BINARY_FORMATS.includes(ext)) {
            // Binary file — send to server for parsing
            parseBinaryFile(file)
                .then(fileValues => applyMatrixFileValues(fileValues))
                .catch(error => alert('Error parsing file: ' + error.message));
        } else {
            // Text file — parse client-side
            const reader = new FileReader();
            reader.onload = function (event) {
                try {
                    const fileValues = parseTextFile(file, event.target.result);
                    applyMatrixFileValues(fileValues);
                } catch (error) {
                    alert('Invalid file format: ' + error.message);
                }
            };
            reader.readAsText(file);
        }
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
    const selectedSolver = document.getElementById('solver').value;

    if (selectedSolver === 'MOSEK') {
        // Use cached license if available, otherwise show the modal
        var cachedLicense = getCachedMosekLicense();
        if (cachedLicense) {
            executeSolve(cachedLicense);
            return;
        }
        $('#mosekLicenseModal').modal('show');
        return;
    }

    executeSolve(null);
}

function executeSolve(mosekLicenseFile) {
    const activeTab = document.querySelector('.nav-link.active').id;
    let formId;
    if (activeTab === 'lp-tab') {
        formId = 'lp-form';
    } else if (activeTab === 'qp-tab') {
        formId = 'qp-form';
    } else if (activeTab === 'sdp-tab') {
        formId = 'sdp-form';
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

    // Append MOSEK license if provided
    if (mosekLicenseFile) {
        formData.append('mosek_license', mosekLicenseFile);
    }

    document.getElementById('result-box').innerHTML = '<p>Solve button pressed, processing results...</p>';

    fetch(activeForm.action, {
        method: activeForm.method,
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error('Server error (' + response.status + '): ' + text);
                });
            }
            return response.json();
        })
        .then(data => {
            // Check for top-level error from server
            if (data.error) {
                document.getElementById('result-box').innerHTML =
                    '<div class="alert alert-danger"><strong>Error:</strong> ' + data.error + '</div>';
                return;
            }
            lastResultData = data;
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
            <h3><br> Optimization Results
                <button class="btn btn-outline-success btn-sm ml-3" onclick="downloadResultsJSON()">Download .JSON</button>
                <button class="btn btn-outline-success btn-sm ml-1" onclick="downloadResultsMAT()">Download .MAT</button>
            </h3>
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
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result-box').innerHTML =
                '<div class="alert alert-danger"><strong>Error:</strong> ' + error.message + '</div>';
        });
}

function generateResultTable(data) {
    // Get arrays of rho and tau
    const rhos = Array.isArray(data.rho_) ? data.rho_ : [data.rho_];
    const taus = Array.isArray(data.tau_) ? data.tau_ : [data.tau_];
    const count = Math.max(rhos.length, taus.length);

    // Generate headers
    const headers = Array.from({length: count}, (_, i) => `<th>Value${i + 1}</th>`).join('');

    // Helper to get value or empty string
    function getValue(arr, idx) {
        if (Array.isArray(arr)) {
            return arr[idx] !== undefined ? arr[idx] : '';
        }
        return arr !== undefined ? arr : '';
    }

    // Only show zeta row when relaxation is active
    const hasRelaxation = rhos.some(r => r > 0);

    // Prepare rows
    const rows = [
        {label: 'Optimal Cost', values: data.optimal_cost},
        {label: `Optimal x`, values: data.optimal_x},
        ...(hasRelaxation ? [{label: 'Optimal &zeta;', values: data.optimal_s}] : []),
        {label: 'Relaxation Parameters &rho;', values: data.rho_},
        {label: 'Regularization Parameters &tau;', values: data.tau_},
        {label: 'Confidence \\(1-\\frac{\\beta}{n_{\\tau} n_{\\rho}}\\)', values: data.conf},
        {label: 'Risk Bounds &epsilon;', values: data.risk},
        {label: 'Degeneracy Detected?', values: data.degeneracy},
        {label: 'Complexity (support list size)', values: data.active_con},
        {label: 'Number of data samples', values: data.num_deltas},
        {label: 'Total Constraints', values: data.tot_con},
        {label: 'Optimization Time (s)', values: data.solve_time},
        {label: 'Risk Computation Time (s)', values: data.risk_time},
        {label: '<i>Error</i>', values: data.errorcode}
    ];

    let table = `<table class="result-table">
        <tr>
            <th>Parameter</th>
            ${headers}
        </tr>`;

    rows.forEach(row => {
        table += `<tr><td>${row.label}</td>`;
        for (let i = 0; i < count; i++) {
            let val = getValue(row.values, i);

            if (row.label === 'Optimal x' || row.label === 'Optimal &zeta;') {
                if (Array.isArray(val)) {
                    val = `<div style="max-height:150px; overflow-y:auto;">${val.join('<br>')}</div>`;
                }
            }

            if (row.label.includes('Risk Bounds') && Array.isArray(val) && val.length === 2) {
                val = `<strong>[${val[0]}, ${val[1]}]</strong>`;
            }
            if (row.label.includes('Optimal Cost')) {
                val = `<strong>${val}</strong>`;
            }
            if (row.label.includes('Error')) {
                val = `<i>${Array.isArray(val) ? val.join('<br>') : val}</i>`;
            }
            if (row.label.includes('Degeneracy Detected?')) {
                val = val ? 'Yes' : 'No';
            }
            table += `<td>${val !== undefined ? val : ''}</td>`;
        }
        table += `</tr>`;
    });
    table += `</table>`;
    return table;
}

function downloadResultsJSON() {
    if (!lastResultData) return;
    // Build a clean results object (exclude form_data for cleanliness)
    var results = Object.assign({}, lastResultData);
    delete results.form_data;
    var blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'results.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function downloadResultsMAT() {
    if (!lastResultData) return;
    var results = Object.assign({}, lastResultData);
    delete results.form_data;
    fetch('/download-mat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(results)
    })
    .then(function(response) {
        if (!response.ok) throw new Error('Failed to generate MAT file');
        return response.blob();
    })
    .then(function(blob) {
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'results.mat';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    })
    .catch(function(err) {
        alert('Error downloading MAT file: ' + err.message);
    });
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
            latexText1 = '\\displaystyle\\min_{x} \\quad c^\\top x';
            latexText4 = 'A(\\delta_i)x+b(\\delta_i) \\leq 0';
            latexText3 = 'i = 1, \\ldots, N.';
            latexText2 = 'Gx+h \\leq 0';
        } else if (option === 'regularization') {
            latexText1 = '\\displaystyle\\min_{x} \\quad c^\\top x + \\tau\\Vert x - \\bar{x}\\Vert_{p}';
            latexText4 = 'A(\\delta_i)x+b(\\delta_i) \\leq 0';
            latexText3 = '\\tau\\geq 0, \\text{and}~ i = 1, \\ldots, N.';
            latexText2 = 'Gx+h \\leq 0';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        } else if (option === 'relaxation') {
            latexText1 = '\\displaystyle\\min_{x,\\zeta_i} \\quad c^\\top x + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText4 = 'A(\\delta_i)x+b(\\delta_i) \\leq \\zeta_i,';
            latexText3 = '\\zeta_i \\geq 0, \\rho\\geq 0, \\text{and}~ i = 1, \\ldots, N.';
            latexText2 = 'Gx+h \\leq 0';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
        } else if (option === 'regularization-relaxation') {
            latexText1 = '\\displaystyle\\min_{x,\\zeta_i} \\quad c^\\top x + \\tau\\Vert x - \\bar{x}\\Vert_{p} + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText4 = 'A(\\delta_i)x+b(\\delta_i) \\leq \\zeta_i';
            latexText3 = '\\zeta_i \\geq 0, \\rho\\geq 0, \\tau\\geq 0, \\text{and}~ i = 1, \\ldots, N.';
            latexText2 = 'Gx+h \\leq 0';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        }
    } else if (tab === 'qp') {
        if (option === 'robust') {
            latexText1 = '\\displaystyle\\min_{x} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx';
            latexText4 = 'A(\\delta_i)x+b(\\delta_i) \\leq 0';
            latexText3 = 'i = 1, \\ldots, N.';
            latexText2 = 'Gx+h \\leq 0';
        } else if (option === 'regularization') {
            latexText1 = '\\displaystyle\\min_{x} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx + \\tau\\Vert x - \\bar{x}\\Vert_{p}';
            latexText4 = 'A(\\delta_i)x+b(\\delta_i) \\leq 0';
            latexText3 = '\\tau\\geq 0, \\text{and}~ i = 1, \\ldots, N.';
            latexText2 = 'Gx+h \\leq 0';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        } else if (option === 'relaxation') {
            latexText1 = '\\displaystyle\\min_{x,\\zeta_i} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText4 = 'A(\\delta_i)x+b(\\delta_i) \\leq \\zeta_i';
            latexText3 = '\\zeta_i \\geq 0, \\rho\\geq 0, \\text{and}~ i = 1, \\ldots, N.';
            latexText2 = 'Gx+h \\leq 0';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
        } else if (option === 'regularization-relaxation') {
            latexText1 = '\\displaystyle\\min_{x,\\zeta_i} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx + \\tau\\Vert x - \\bar{x}\\Vert_{p} + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText4 = 'A(\\delta_i)x+b(\\delta_i) \\leq \\zeta_i';
            latexText3 = '\\zeta_i \\geq 0, \\rho\\geq 0, \\tau\\geq 0, \\text{and}~ i = 1, \\ldots, N.';
            latexText2 = 'Gx+h \\leq 0';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        }
    } else if (tab === 'sdp') {
        if (option === 'robust') {
            latexText1 = '\\displaystyle\\min_{x} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx';
            latexText4 = 'F_0(\\delta_i) + \\displaystyle\\sum_{j=1}^{d}x_jF_j(\\delta_i) \\leq 0';
            latexText3 = ' i = 1, \\ldots, N.';
            latexText2 = 'E_0 + \\displaystyle\\sum_{j=1}^{d}x_jE_j \\leq 0';
        } else if (option === 'regularization') {
            latexText1 = '\\displaystyle\\min_{x} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx + \\tau\\Vert x - \\bar{x}\\Vert_{p}';
            latexText4 = 'F_0(\\delta_i) + \\displaystyle\\sum_{j=1}^{d}x_jF_j(\\delta_i) \\leq 0';
            latexText3 = '\\tau\\geq 0, \\text{and}~ i = 1, \\ldots, N.';
            latexText2 = 'E_0 + \\displaystyle\\sum_{j=1}^{d}x_jE_j \\leq 0';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        } else if (option === 'relaxation') {
            latexText1 = '\\displaystyle\\min_{x,\\zeta_i} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText4 = 'F_0(\\delta_i) + \\displaystyle\\sum_{j=1}^{d}x_jF_j(\\delta_i) \\leq \\zeta_i';
            latexText3 = '\\zeta_i \\geq 0, \\rho\\geq 0, \\text{and}~ i = 1, \\ldots, N.';
            latexText2 = 'E_0 + \\displaystyle\\sum_{j=1}^{d}x_jE_j \\leq 0';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
        } else if (option === 'regularization-relaxation') {
            latexText1 = '\\displaystyle\\min_{x,\\zeta_i}\\quad c^\\top x + \\frac{1}{2}x^\\top Qx + \\tau\\Vert x - \\bar{x}\\Vert_{p} + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexText4 = 'F_0(\\delta_i) + \\displaystyle\\sum_{j=1}^{n}x_jF_j(\\delta_i) \\leq \\zeta_i';
            latexText3 = '\\zeta_i \\geq 0, \\rho\\geq 0, \\tau\\geq 0, \\text{and}~ i = 1, \\ldots, N.';
            latexText2 = 'E_0 + \\displaystyle\\sum_{j=1}^{n}x_jE_j \\leq 0';
            document.getElementById(`${tab}-tau-group`).style.display = 'block';
            document.getElementById(`${tab}-theta-bar-group`).style.display = 'block';
            document.getElementById(`${tab}-rho-group`).style.display = 'block';
            document.getElementById(`${tab}-p-group`).style.display = 'block';
        }
    }

    document.getElementById(`${tab}-latex`).innerHTML = `<div class="text-center"><p>\\(${latexText1}\\)</p><p>subject to:</p><p>\\(${latexText4}\\)</p><p>\\(${latexText2}\\)</p><p>where \\(${latexText3}\\)</p></div>`;
    MathJax.typeset();
}

function loadProblemJSON() {
    const fileInput = document.getElementById('detect-program-file');
    const textarea = document.getElementById('detect-program-text');

    function processData(data) {

        const type = (data.type || '').toUpperCase();
        if (!['LP', 'QP', 'SDP'].includes(type)) {
            alert('JSON must include a "type" field with value "LP", "QP", or "SDP".');
            return;
        }

        // Switch to the correct tab
        const tabId = type.toLowerCase() + '-tab';
        document.getElementById(tabId).click();

        // Determine the formulation option based on rho/tau
        const hasRho = data.rho !== undefined && parseFloat(data.rho) > 0;
        const hasTau = data.tau !== undefined && parseFloat(data.tau) > 0;
        let option = 'robust';
        if (hasRho && hasTau) option = 'regularization-relaxation';
        else if (hasRho) option = 'relaxation';
        else if (hasTau) option = 'regularization';

        const prefix = type.toLowerCase();
        const optionSelect = document.getElementById(prefix + '-options');
        if (optionSelect) {
            optionSelect.value = option;
            updateLatexText(prefix);
        }

        if (type === 'LP' || type === 'QP') {
            // Set matrices: A_d, b_d, G(=A), h(=b), c
            if (data.A_d) document.getElementById(prefix + '-A_d').value = JSON.stringify(data.A_d);
            if (data.b_d) document.getElementById(prefix + '-b_d').value = JSON.stringify(data.b_d);
            if (data.G) document.getElementById(prefix + '-A').value = JSON.stringify(data.G);
            if (data.h) document.getElementById(prefix + '-b').value = JSON.stringify(data.h);
            if (data.c) document.getElementById(prefix + '-c').value = JSON.stringify(data.c);
            if (type === 'QP' && data.Q) {
                document.getElementById('qp-Q').value = JSON.stringify(data.Q);
            }
        } else if (type === 'SDP') {
            // F_d is a dict of expression matrices keyed by "0","1",...
            if (data.F_d) document.getElementById('sdp-F_d').value = JSON.stringify(data.F_d);
            // E is the hard constraint dict
            if (data.E) document.getElementById('sdp-F').value = JSON.stringify(data.E);
            if (data.c) document.getElementById('sdp-c').value = JSON.stringify(data.c);
            if (data.Q) document.getElementById('sdp-Q').value = JSON.stringify(data.Q);
        }

        // Set parameters
        if (data.rho !== undefined) {
            const rhoInput = document.getElementById(prefix + '-rho');
            if (rhoInput) rhoInput.value = data.rho;
        }
        if (data.tau !== undefined) {
            const tauInput = document.getElementById(prefix + '-tau');
            if (tauInput) tauInput.value = data.tau;
        }
        if (data.confidence !== undefined) {
            document.getElementById('confidence').value = data.confidence;
        }
        if (data.p !== undefined) {
            const pInput = document.getElementById(prefix + '-p');
            if (pInput) pInput.value = data.p;
        }
        if (data.x_ref !== undefined) {
            const thetaInput = document.getElementById(prefix + '-theta-bar');
            if (thetaInput) thetaInput.value = JSON.stringify(data.x_ref);
        }

        // Close modal and notify
        $('#detectProgramModal').modal('hide');
        // Clear inputs for next use
        fileInput.value = '';
        textarea.value = '';
        alert(type + ' problem loaded successfully. Upload scenarios and press Solve.');
    }

    // File takes priority over textarea
    if (fileInput && fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const ext = file.name.split('.').pop().toLowerCase();

        if (ext === 'mat') {
            // MAT file — send to server for parsing
            parseBinaryFile(file)
                .then(data => processData(data))
                .catch(error => alert('Error parsing MAT file: ' + error.message));
        } else {
            // JSON file — parse client-side
            const reader = new FileReader();
            reader.onload = function(e) {
                let data;
                try {
                    data = JSON.parse(e.target.result);
                } catch (err) {
                    alert('Invalid JSON: ' + err.message);
                    return;
                }
                processData(data);
            };
            reader.readAsText(file);
        }
    } else if (textarea.value.trim()) {
        let data;
        try {
            data = JSON.parse(textarea.value.trim());
        } catch (e) {
            alert('Invalid JSON: ' + e.message);
            return;
        }
        processData(data);
    } else {
        alert('Please upload a file or paste JSON content.');
    }
}

function filterSolvers() {
    const activeTab = document.querySelector('.nav-link.active');
    if (!activeTab) return;

    const tabTypeMap = {'lp-tab': 'LP', 'qp-tab': 'QP', 'sdp-tab': 'SDP'};
    const problemType = tabTypeMap[activeTab.id];
    if (!problemType) return;

    const solverSelect = document.getElementById('solver');
    const currentValue = solverSelect.value;
    let firstVisible = null;

    Array.from(solverSelect.options).forEach(option => {
        const types = (option.getAttribute('data-types') || '').split(',');
        if (types.includes(problemType)) {
            option.style.display = '';
            option.disabled = false;
            if (!firstVisible) firstVisible = option.value;
        } else {
            option.style.display = 'none';
            option.disabled = true;
        }
    });

    // If current selection is now hidden, switch to first visible solver
    const currentOption = solverSelect.querySelector('option[value="' + currentValue + '"]');
    if (currentOption && currentOption.disabled && firstVisible) {
        solverSelect.value = firstVisible;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Filter solvers on tab change
    $('a[data-toggle="tab"]').on('shown.bs.tab', function() {
        filterSolvers();
    });

    // Initial filter on page load
    filterSolvers();

    // Initialize MOSEK cache indicator
    updateMosekCacheIndicator();

    // MOSEK license modal: upload & solve (cache for future use)
    document.getElementById('mosek-license-confirm').addEventListener('click', function() {
        const fileInput = document.getElementById('mosek-license-file');
        if (!fileInput.files.length) {
            alert('Please select a MOSEK license file.');
            return;
        }
        const licenseFile = fileInput.files[0];
        $('#mosekLicenseModal').modal('hide');
        fileInput.value = '';
        cacheMosekLicense(licenseFile).then(function() {
            executeSolve(licenseFile);
        });
    });

    // MOSEK license modal: skip (local license installed)
    document.getElementById('mosek-license-skip').addEventListener('click', function() {
        document.getElementById('mosek-license-file').value = '';
        $('#mosekLicenseModal').modal('hide');
        executeSolve(null);
    });
});