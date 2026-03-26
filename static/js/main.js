window.sdpMatrixCollection1 = {};
window.sdpMatrixCollection2 = {};
window.sdpMatrixCollection3 = {};
window.sdpMatrixCollection4 = {};
window.sdpCurrentMatrixIndex = null;
let currentMatrix = '';
let lastResultData = null;
let currentMode = 'symbolic';

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
    const n = parseInt(document.getElementById('dim-nx').value, 10) || 2;
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

    const sdpFileInput = document.getElementById('SDP-file');
    sdpFileInput.value = '';
    // Both soft (delta) and hard collections support JSON and MAT
    const isSoft = ['F_d', 'A_da', 'b_da'].includes(matrix);
    sdpFileInput.accept = '.json,.mat';
    sdpFileInput.nextElementSibling.textContent = isSoft
        ? 'Supported: JSON, MAT (cell arrays of strings for expressions)'
        : 'Supported: JSON, MAT (dict of matrices keyed by index)';

    // Create dynamic buttons
    updateSDPButtons();

    // Set default dimensions based on the matrix
    var titleMap = {
        'F_d': `Edit \\(F_j(\\delta)\\)`, 'F': `Edit \\(E_j\\)`,
        'A_da': `Edit \\(A_j(\\delta)\\)`, 'A_a': `Edit \\(G_j\\)`,
        'b_da': `Edit \\(b_j(\\delta)\\)`, 'b_a': `Edit \\(h_j\\)`
    };
    if (titleMap[matrix]) {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0, 0]];
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `${titleMap[matrix]} (${tab.toUpperCase()})`;
        document.getElementById('sdpModalTip').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        updateSDPFormatReference(matrix);
        MathJax.typesetPromise();
        $('#SDPModal').modal('show');
    }
}

function openSDPMatrixEditor(idx) {

    var collectionMap = {
        'F_d': 'sdpMatrixCollection1', 'A_da': 'sdpMatrixCollection1',
        'F': 'sdpMatrixCollection2', 'A_a': 'sdpMatrixCollection2',
        'b_da': 'sdpMatrixCollection3', 'b_a': 'sdpMatrixCollection4'
    };
    var suffix = Object.keys(collectionMap).find(function(s) { return currentMatrix.endsWith(s); });
    if (suffix) {
        var collName = collectionMap[suffix];
        if (!window[collName]) window[collName] = {};
        window.sdpCurrentMatrixIndex = idx;
        // F matrices use mathfrak{m} (dim-lmi-size), E matrices use mathfrak{n} (dim-lmi-e-size)
        var isHardConstraint = (suffix === 'F' || suffix === 'A_a' || suffix === 'b_a');
        var sizeInput = isHardConstraint ? 'dim-lmi-e-size' : 'dim-lmi-size';
        var sdpCols = parseInt(document.getElementById(sizeInput).value) || 2;
        var values = window[collName][idx] ||
            Array.from({length: sdpCols}, function() { return Array(sdpCols).fill(0); });
        updateMatrixGrid(values);
        document.getElementById('matrixModalTitle').textContent = 'Edit Matrix ' + idx;
        $('#matrixModal').modal('show');
    }
}

function getDimDefaults(matrix) {
    const nx = parseInt(document.getElementById('dim-nx').value) || 2;
    const rowsA = parseInt(document.getElementById('dim-rows-a').value) || 1;
    const rowsG = parseInt(document.getElementById('dim-rows-g').value) || 0;
    const lmi = parseInt(document.getElementById('dim-lmi-size').value) || 2;

    const zeros = (r, c) => Array.from({length: r}, () => Array(c).fill(0));

    switch (matrix) {
        case 'c':       return {rows: nx, cols: 1,   colDisabled: true,  vals: zeros(nx, 1)};
        case 'Q':       return {rows: nx, cols: nx,  colDisabled: false, vals: zeros(nx, nx)};
        case 'A_d':     return {rows: rowsA, cols: nx, colDisabled: false, vals: zeros(rowsA, nx)};
        case 'b_d':     return {rows: rowsA, cols: 1,  colDisabled: true,  vals: zeros(rowsA, 1)};
        case 'A':       return {rows: Math.max(rowsG, 1), cols: nx, colDisabled: false, vals: zeros(Math.max(rowsG, 1), nx)};
        case 'b':       return {rows: Math.max(rowsG, 1), cols: 1,  colDisabled: true,  vals: zeros(Math.max(rowsG, 1), 1)};
        case 'theta-bar': return {rows: nx, cols: 1, colDisabled: true, vals: zeros(nx, 1)};
        case 'C':       return {rows: lmi, cols: lmi, colDisabled: false, vals: zeros(lmi, lmi)};
        case 'Theta-bar': return {rows: nx, cols: nx, colDisabled: false, vals: zeros(nx, nx)};
        default:        return {rows: 2, cols: 2, colDisabled: false, vals: zeros(2, 2)};
    }
}

function openMatrixModal(tab, matrix) {
    currentMatrix = `${tab}-${matrix}`;
    const matrixInput = document.getElementById(currentMatrix);
    const def = getDimDefaults(matrix);

    const fileInput = document.getElementById('modal-file');
    fileInput.value = '';
    // Restrict upload formats: delta matrices accept text formats + MAT (cell arrays)
    const isDelta = ['A_d', 'b_d'].includes(matrix);
    fileInput.accept = isDelta
        ? '.csv,.txt,.tsv,.json,.mat'
        : '.csv,.txt,.tsv,.json,.npy,.npz,.mat,.xlsx,.xls,.parquet';
    fileInput.nextElementSibling.textContent = isDelta
        ? 'Supported: CSV, TXT, TSV, JSON, MAT (cell arrays)'
        : 'Supported: CSV, JSON, TXT, TSV, MAT, Excel, NPY, Parquet';

    let matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : def.vals;
    updateMatrixGrid(matrixValues);

    const titles = {
        'A_d': `Edit A(\\(\\delta)\\)`, 'b_d': `Edit b(\\(\\delta)\\)`,
        'c': 'Edit c', 'Q': 'Edit Q', 'A': 'Edit G', 'b': 'Edit h',
        'C': 'Edit C', 'theta-bar': `Edit \\(\\bar{x}\\)`,
        'Theta-bar': `Edit \\(\\bar{X}\\)`
    };
    document.getElementById('matrixModalTitle').textContent =
        `${titles[matrix] || 'Edit ' + matrix} (${tab.toUpperCase()})`;
    document.getElementById('matrixModalTip').innerHTML =
        `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
    updateFormatReference(matrix);
    MathJax.typesetPromise();
    $('#matrixModal').modal('show');
}

function getDynamicTip(matrix) {
    if (matrix === 'A_d') {
        return "Manually enter or upload the matrix. Cells accept SymPy expressions with <code>delta[i]</code> (delta[0] = \\(\\delta_1\\), delta[1] = \\(\\delta_2\\), ...) including +, -, *, /, **.";
    } else if (matrix === 'b_d') {
        return "Manually enter or upload the vector. Cells accept SymPy expressions with <code>delta[i]</code> (delta[0] = \\(\\delta_1\\), delta[1] = \\(\\delta_2\\), ...) including +, -, *, /, **.";
    } else if (matrix === 'Q') {
        return "Manually enter or upload the matrix. \\(Q\\) must be <b>symmetric</b> and <b>positive semidefinite</b> (\\(Q \\succeq 0\\)).";
    } else if (matrix === 'c') {
        return "Manually enter or upload the cost vector.";
    } else if (matrix === 'A') {
        return "Manually enter or upload the hard constraint matrix.";
    } else if (matrix === 'b') {
        return "Manually enter or upload the hard constraint vector.";
    } else if (matrix === 'F_d') {
        return "Manually enter or upload the matrices. Each \\(F_j(\\delta)\\) must be <b>symmetric</b>. Cells accept SymPy expressions with <code>delta[i]</code> (delta[0] = \\(\\delta_1\\), delta[1] = \\(\\delta_2\\), ...) including +, -, *, /, **.";
    } else if (matrix === 'F') {
        return "Manually enter or upload the matrices. Each \\(E_j\\) must be <b>symmetric</b>.";
    } else if (matrix === 'A_da') {
        return "Manually enter or upload the matrices. Each \\(A_j(\\delta)\\) must be <b>symmetric</b>. Cells accept SymPy expressions with <code>delta[i]</code> (delta[0] = \\(\\delta_1\\), delta[1] = \\(\\delta_2\\), ...) including +, -, *, /, **.";
    } else if (matrix === 'A_a') {
        return "Manually enter or upload the matrices. Each \\(G_j\\) must be <b>symmetric</b>.";
    } else if (matrix === 'C') {
        return "Manually enter or upload the matrix. \\(C\\) must be <b>symmetric</b> and <b>positive semidefinite</b>.";
    } else if (matrix === 'Theta-bar') {
        return "Manually enter or upload the reference matrix.";
    } else if (matrix === 'theta-bar') {
        return "Manually enter or upload the reference vector.";
    } else if (matrix === 'b_da') {
        return "Manually enter or upload the vectors. Cells accept SymPy expressions with <code>delta[i]</code> (delta[0] = \\(\\delta_1\\), delta[1] = \\(\\delta_2\\), ...) including +, -, *, /, **.";
    } else if (matrix === 'b_a') {
        return "Manually enter or upload the vectors.";
    }
    return "Manually enter or upload the data.";
}

/* ── Bespoke Input Format Reference for the matrix modal ── */
function updateFormatReference(matrix) {
    const ref = document.getElementById('matrixFormatRef');
    if (!ref) return;

    const pre = 'style="background:#f8f9fa; padding:6px; border-radius:4px; font-size:0.85em; margin:4px 0;"';

    // Determine category
    const isDelta = ['A_d', 'b_d'].includes(matrix);
    const isQ = (matrix === 'Q');
    const exName = isQ ? 'Q' : matrix === 'c' ? 'c' : matrix === 'A' ? 'G' : matrix === 'b' ? 'h' : matrix;

    // ── Manual entry ──
    let manual = '<h6><b>Manual entry:</b></h6>';
    if (isDelta) {
        manual += '<p>The grid is auto-sized from the dimension controls (\\(d\\), \\(\\mathfrak{m}\\)). ' +
                  'Cells accept numbers or any SymPy expressions with <code>delta[i]</code>, ' +
                  'including operators +, -, *, /, ** and functions like <code>sin()</code>, <code>exp()</code> ' +
                  '(e.g. <code>delta[0] + 1</code>, <code>sin(delta[1])</code>).</p>';
    } else if (isQ) {
        manual += '<p>The grid is auto-sized from the dimension control \\(d\\). ' +
                  'Cells accept numeric values. \\(Q\\) must be symmetric and positive semidefinite (\\(Q \\succeq 0\\)).</p>';
    } else {
        manual += '<p>The grid is auto-sized from the dimension controls. Cells accept numeric values.</p>';
    }

    // ── File formats ──
    let formats = '<h6><b>Supported file formats:</b></h6><ul>';
    // Text-capable formats (always listed)
    formats += `<li><b>CSV/TXT:</b> Comma-separated values, one row per line.
        <pre ${pre}>${isDelta ? 'delta[0]+1, 0\n0, delta[1]' : '1.0, 0\n0, -1.5'}</pre></li>`;
    formats += '<li><b>TSV:</b> Same as CSV but tab-separated.</li>';
    formats += `<li><b>JSON:</b> 2D array of ${isDelta ? 'strings or numbers' : 'numbers'}.
        <pre ${pre}>${isDelta ? '[["delta[0]+1", "0"], ["0", "delta[1]"]]' : '[[1.0, 0], [0, -1.5]]'}</pre></li>`;
    // MAT: always supported (cell arrays for expressions, numeric arrays otherwise)
    if (isDelta) {
        formats += `<li><b>MAT:</b> MATLAB <code>.mat</code> file. Use a <b>cell array of strings</b> for expressions.
            <pre ${pre}>% MATLAB: ${exName} = {'delta[0]+1', '0'; '0', 'delta[1]'};
% save('${exName}.mat', '${exName}')</pre></li>`;
    } else {
        formats += `<li><b>MAT:</b> MATLAB <code>.mat</code> file. The first non-metadata variable is used.
            <pre ${pre}>% MATLAB: save('${exName}.mat', '${exName}')
% where ${exName} = [1 0; 0 -1]</pre></li>`;
    }
    // Binary formats (numeric only)
    if (!isDelta) {
        formats += '<li><b>Excel:</b> <code>.xlsx</code> / <code>.xls</code> file. Reads the first sheet with no header row.</li>';
        formats += `<li><b>NPY:</b> NumPy <code>.npy</code> file containing a 2D array.
            <pre ${pre}># Python: np.save('${exName}.npy', ${exName})</pre></li>`;
        formats += '<li><b>NPZ:</b> NumPy <code>.npz</code> archive. The first array is used.</li>';
        formats += '<li><b>Parquet:</b> Apache Parquet file. Columns become matrix columns; rows become rows.</li>';
    }
    formats += '</ul>';

    ref.innerHTML = manual + formats;
    MathJax.typesetPromise([ref]);
}

/* ── Bespoke Input Format Reference for the SDP collection modal ── */
function updateSDPFormatReference(matrix) {
    const ref = document.getElementById('sdpFormatRef');
    if (!ref) return;

    const pre = 'style="background:#f8f9fa; padding:6px; border-radius:4px; font-size:0.85em; margin:4px 0;"';
    const isSoft = ['F_d', 'A_da', 'b_da'].includes(matrix);  // delta-capable collections

    let html = '';

    // ── Individual entry ──
    html += '<h6><b>Individual entry:</b></h6>';
    if (isSoft) {
        html += '<p>Click each matrix button to open an editor. Each matrix should be symmetric. ' +
                'Cells accept numbers or any SymPy expressions with <code>delta[i]</code>, ' +
                'including operators +, -, *, /, ** and functions like <code>sin()</code>, <code>exp()</code>.</p>';
    } else {
        html += '<p>Click each matrix button to open an editor. Each matrix should be symmetric. ' +
                'Cells accept numeric values.</p>';
    }

    // ── Collection upload ──
    html += '<h6><b>File upload (all at once):</b></h6>';
    if (isSoft) {
        html += '<p>Supported: <b>JSON</b> and <b>MAT</b> (using cell arrays of strings for expressions).</p>';
        html += '<ul>';
        html += `<li><b>JSON:</b> Dictionary keyed by index ("0", "1", ...), each value a 2D array of strings.
            <pre ${pre}>{"0": [["delta[0]", "0"], ["0", "1"]],
 "1": [["1", "0"], ["0", "-1"]], ...}</pre></li>`;
        html += `<li><b>MAT:</b> MATLAB <code>.mat</code> file. Use <b>cell arrays of strings</b> for expressions.
            <pre ${pre}>% MATLAB: F0 = {'delta[0]', '0'; '0', '1'};
% F1 = {'1', '0'; '0', '-1'};
% save('F_d.mat', 'F0', 'F1')</pre></li>`;
        html += '</ul>';
    } else {
        html += '<p>Supported: <b>JSON</b> and <b>MAT</b>.</p>';
        html += '<ul>';
        html += `<li><b>JSON:</b> Dictionary keyed by index ("0", "1", ...), each value a 2D numeric array.
            <pre ${pre}>{"0": [[1, 0], [0, 1]],
 "1": [[0, 1], [1, 0]], ...}</pre></li>`;
        html += `<li><b>MAT:</b> MATLAB <code>.mat</code> file where each variable is one matrix.
            <pre ${pre}>% MATLAB: save('E.mat', 'E0', 'E1', 'E2')
% Variable names become dict keys</pre></li>`;
        html += '</ul>';
    }

    ref.innerHTML = html;
    MathJax.typesetPromise([ref]);
}

function updateMatrixGrid(values = null, numRows = null, numCols = null) {
    const rows = numRows || (values ? values.length : 2);
    const columns = numCols || (values && values[0] ? values[0].length : 2);
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
    const grid = document.getElementById('matrixGrid');
    const rowDivs = grid.querySelectorAll('.matrix-row');
    const rows = rowDivs.length;
    const columns = rows > 0 ? rowDivs[0].querySelectorAll('input').length : 0;
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

    // Append mode and dimension controls
    formData.append('mode', currentMode);
    formData.append('n_x', document.getElementById('dim-nx').value);
    formData.append('rows_A', document.getElementById('dim-rows-a').value);
    formData.append('rows_G', document.getElementById('dim-rows-g').value);
    formData.append('lmi_size', document.getElementById('dim-lmi-size').value);

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
    const isNumeric = currentMode === 'numeric';
    let latexObj = ''; // objective
    let latexSoft = ''; // soft constraint with where-clause
    let latexHard = ''; // hard constraint

    // Show/Hide relevant input fields (toggle the parent column so hidden
    // params don't occupy space in the row)
    const hideParam = id => {
        const el = document.getElementById(`${tab}-${id}`);
        el.style.display = 'none';
        el.parentElement.style.display = 'none';
    };
    const showParam = id => {
        const el = document.getElementById(`${tab}-${id}`);
        el.style.display = 'block';
        el.parentElement.style.display = '';
    };
    hideParam('tau-group');
    hideParam('theta-bar-group');
    hideParam('rho-group');
    hideParam('p-group');

    // LP / QP constraint notation
    const Ax = isNumeric ? 'A_{i}\\, x + b_{i}' : 'A(\\delta_i)\\, x+b(\\delta_i)';
    // SDP constraint notation
    const Fsum = isNumeric
        ? 'F_{0,i} + \\displaystyle\\sum_{j=1}^{d}x_j F_{j,i}'
        : 'F_0(\\delta_i) + \\displaystyle\\sum_{j=1}^{d}x_j F_j(\\delta_i)';

    const showTau = () => {
        showParam('tau-group');
        showParam('theta-bar-group');
        showParam('p-group');
    };
    const showRho = () => {
        showParam('rho-group');
    };

    if (tab === 'lp' || tab === 'qp') {
        const qTerm = tab === 'qp' ? ' + \\frac{1}{2}x^\\top Qx' : '';
        latexHard = 'Gx+h \\leq 0';
        if (option === 'robust') {
            latexObj = `\\displaystyle\\min_{x} \\quad c^\\top x${qTerm}`;
            latexSoft = `${Ax} \\leq 0, \\quad i = 1, \\ldots, N`;
        } else if (option === 'regularization') {
            latexObj = `\\displaystyle\\min_{x} \\quad c^\\top x${qTerm} + \\tau\\Vert x - \\bar{x}\\Vert_{p}`;
            latexSoft = `${Ax} \\leq 0, \\quad \\tau\\geq 0,~ i = 1, \\ldots, N`;
            showTau();
        } else if (option === 'relaxation') {
            latexObj = `\\displaystyle\\min_{x,\\zeta_i} \\quad c^\\top x${qTerm} + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i`;
            latexSoft = `${Ax} \\leq \\zeta_i, \\quad \\zeta_i \\geq 0,~ \\rho\\geq 0,~ i = 1, \\ldots, N`;
            showRho();
        } else if (option === 'regularization-relaxation') {
            latexObj = `\\displaystyle\\min_{x,\\zeta_i} \\quad c^\\top x${qTerm} + \\tau\\Vert x - \\bar{x}\\Vert_{p} + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i`;
            latexSoft = `${Ax} \\leq \\zeta_i, \\quad \\zeta_i \\geq 0,~ \\rho\\geq 0,~ \\tau\\geq 0,~ i = 1, \\ldots, N`;
            showTau(); showRho();
        }
    } else if (tab === 'sdp') {
        latexHard = 'E_0 + \\displaystyle\\sum_{j=1}^{d}x_jE_j \\preceq 0';
        if (option === 'robust') {
            latexObj = '\\displaystyle\\min_{x} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx';
            latexSoft = `${Fsum} \\preceq 0, \\quad i = 1, \\ldots, N`;
        } else if (option === 'regularization') {
            latexObj = '\\displaystyle\\min_{x} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx + \\tau\\Vert x - \\bar{x}\\Vert_{p}';
            latexSoft = `${Fsum} \\preceq 0, \\quad \\tau\\geq 0,~ i = 1, \\ldots, N`;
            showTau();
        } else if (option === 'relaxation') {
            latexObj = '\\displaystyle\\min_{x,\\zeta_i} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexSoft = `${Fsum} \\preceq \\zeta_i I, \\quad \\zeta_i \\geq 0,~ \\rho\\geq 0,~ i = 1, \\ldots, N`;
            showRho();
        } else if (option === 'regularization-relaxation') {
            latexObj = '\\displaystyle\\min_{x,\\zeta_i} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx + \\tau\\Vert x - \\bar{x}\\Vert_{p} + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexSoft = `${Fsum} \\preceq \\zeta_i I, \\quad \\zeta_i \\geq 0,~ \\rho\\geq 0,~ \\tau\\geq 0,~ i = 1, \\ldots, N`;
            showTau(); showRho();
        }
    }

    document.getElementById(`${tab}-latex`).innerHTML =
        `<div class="text-center"><p>\\(${latexObj}\\)</p><p>subject to:</p><p>\\(${latexSoft}\\)</p><p>\\(${latexHard}\\)</p></div>`;
    MathJax.typeset();
    updateConfidenceLabel(option);
}

function updateConfidenceLabel(option) {
    if (!option) {
        const activeTab = document.querySelector('.nav-link.active');
        const tab = activeTab ? activeTab.id.replace('-tab', '') : 'lp';
        option = document.getElementById(`${tab}-options`).value;
    }
    const span = document.getElementById('confidence-formula');
    if (option === 'robust') {
        span.innerHTML = '\\(1 - \\beta\\)';
    } else if (option === 'relaxation') {
        span.innerHTML = '\\(1 - \\frac{\\beta}{n_\\rho}\\)';
    } else if (option === 'regularization') {
        span.innerHTML = '\\(1 - \\frac{\\beta}{n_\\tau}\\)';
    } else {
        span.innerHTML = '\\(1 - \\frac{\\beta}{n_\\tau \\cdot n_\\rho}\\)';
    }
    MathJax.typesetPromise();
    updateSweepCounter(option);
}

function updateSweepCounter(option) {
    const counter = document.getElementById('sweep-counter');
    const activeTab = document.querySelector('.nav-link.active');
    const tab = activeTab ? activeTab.id.replace('-tab', '') : 'lp';
    if (!option) option = document.getElementById(`${tab}-options`).value;

    const rhoField = document.getElementById(`${tab}-rho`);
    const tauField = document.getElementById(`${tab}-tau`);
    const nRho = rhoField && rhoField.value ? rhoField.value.split(',').filter(s => s.trim()).length : 1;
    const nTau = tauField && tauField.value ? tauField.value.split(',').filter(s => s.trim()).length : 1;

    const hasRho = option === 'relaxation' || option === 'regularization-relaxation';
    const hasTau = option === 'regularization' || option === 'regularization-relaxation';

    let parts = [];
    if (hasTau) parts.push(`\\(n_\\tau = ${nTau}\\)`);
    if (hasRho) parts.push(`\\(n_\\rho = ${nRho}\\)`);

    if (parts.length > 0) {
        counter.innerHTML = parts.join(', ');
        counter.style.display = 'block';
    } else {
        counter.style.display = 'none';
    }
    MathJax.typesetPromise();
}

function onModeChange() {
    document.querySelectorAll('.symbolic-only').forEach(el => {
        el.style.display = currentMode === 'symbolic' ? '' : 'none';
    });
    updateScenarioLabel();
    MathJax.typesetPromise();
    // Re-render LaTeX for active tab
    const activeTab = document.querySelector('.nav-link.active');
    if (activeTab) {
        updateLatexText(activeTab.id.replace('-tab', ''));
    }
}

function updateModeHelpContent() {
    var activeTab = document.querySelector('.nav-link.active');
    var isSDP = activeTab && activeTab.id === 'sdp-tab';
    var numericDesc = isSDP
        ? '<b>Numeric:</b> Provide pre-computed constraint data directly. Each row of the uploaded dataset contains the row-wise flattened \\(F_{0,i}\\), \\(F_{1,i}\\), \\(\\ldots\\), \\(F_{d,i}\\) matrices concatenated together.'
        : '<b>Numeric:</b> Provide pre-computed constraint data directly. Each row of the uploaded dataset contains the row-wise flattened \\(A_i\\) matrix concatenated with the row-wise flattened \\(b_i\\) vector.';
    var content = '<b>Symbolic:</b> Enter constraint matrices as expressions using <code>delta[i]</code> variables. The matrices are evaluated at each scenario point.<br><br>' + numericDesc;
    var btn = document.getElementById('mode-help-btn');
    $(btn).attr('data-content', content);
    // Update the popover instance if it exists
    var popover = $(btn).data('bs.popover');
    if (popover) {
        popover.config.content = content;
    }
}

function updateScenarioLabel() {
    const label = document.getElementById('scenario-upload-label');
    if (currentMode === 'numeric') {
        const activeTab = document.querySelector('.nav-link.active');
        const isSDP = activeTab && activeTab.id === 'sdp-tab';
        if (isSDP) {
            label.innerHTML = 'Scenario data rows \\([F_{0,i} \\mid F_{1,i} \\mid \\cdots \\mid F_{d,i}]\\):';
        } else {
            label.innerHTML = 'Scenario data rows \\([A_i \\mid b_i]\\):';
        }
    } else {
        label.innerHTML = 'Scenarios \\((\\delta_1, \\delta_2, \\ldots)\\):';
    }
    MathJax.typesetPromise();
    updateScenarioHelpContent();
}

function updateScenarioHelpContent() {
    var activeTab = document.querySelector('.nav-link.active');
    var isSDP = activeTab && activeTab.id === 'sdp-tab';
    var content;
    if (currentMode === 'numeric') {
        if (isSDP) {
            content = 'Upload scenario data as a file. Each row is one scenario sample containing the row-wise flattened matrices \\(F_{0,i},\\, F_{1,i},\\, \\ldots,\\, F_{d,i}\\) concatenated together.';
        } else {
            content = 'Upload scenario data as a file. Each row is one scenario sample containing the row-wise flattened \\(A_i\\) matrix concatenated with the row-wise flattened \\(b_i\\) vector.';
        }
    } else {
        if (isSDP) {
            content = 'Upload scenario data as a file. Each row is one scenario sample containing the uncertain parameter vector \\(\\delta\\) which is substituted into the symbolic expressions for \\(F_j(\\delta)\\).';
        } else {
            content = 'Upload scenario data as a file. Each row is one scenario sample containing the uncertain parameter vector \\(\\delta\\) which is substituted into the symbolic expressions for \\(A(\\delta)\\) and \\(b(\\delta)\\).';
        }
    }
    var btn = document.getElementById('scenario-help-btn');
    $(btn).attr('data-content', content);
    var popover = $(btn).data('bs.popover');
    if (popover) {
        popover.config.content = content;
    }
}

function updateDimensionVisibility() {
    const activeTab = document.querySelector('.nav-link.active');
    const isSDP = activeTab && activeTab.id === 'sdp-tab';
    document.getElementById('dim-rows-a-group').style.display = isSDP ? 'none' : '';
    document.getElementById('dim-rows-g-group').style.display = isSDP ? 'none' : '';
    document.getElementById('dim-lmi-group').style.display = isSDP ? '' : 'none';
    document.getElementById('dim-lmi-e-group').style.display = isSDP ? '' : 'none';
    updateHardConstraintState();
}

function updateHardConstraintState() {
    const activeTab = document.querySelector('.nav-link.active');
    const isSDP = activeTab && activeTab.id === 'sdp-tab';
    var nVal = isSDP
        ? parseInt(document.getElementById('dim-lmi-e-size').value) || 0
        : parseInt(document.getElementById('dim-rows-g').value) || 0;
    var disabled = (nVal === 0);
    document.querySelectorAll('.hard-constraint-btn').forEach(function(btn) {
        btn.disabled = disabled;
        if (disabled) {
            btn.classList.remove('btn-secondary');
            btn.classList.add('btn-outline-secondary');
        } else {
            btn.classList.remove('btn-outline-secondary');
            btn.classList.add('btn-secondary');
        }
    });
}

function loadProblemJSON() {
    const fileInput = document.getElementById('detect-program-file');
    const textarea = document.getElementById('detect-program-text');

    function processData(data) {

        // Clear all previous state before loading a new program
        // Reset SDP matrix collections
        window.sdpMatrixCollection1 = {};
        window.sdpMatrixCollection2 = {};
        window.sdpMatrixCollection3 = {};
        window.sdpMatrixCollection4 = {};
        window.sdpCurrentMatrixIndex = null;

        // Clear all form fields across all tabs
        ['lp', 'qp', 'sdp'].forEach(function(p) {
            ['A_d', 'b_d', 'A', 'b', 'c', 'Q', 'F_d', 'F'].forEach(function(field) {
                var el = document.getElementById(p + '-' + field);
                if (el) el.value = '';
            });
            ['rho', 'tau', 'p'].forEach(function(field) {
                var el = document.getElementById(p + '-' + field);
                if (el) el.value = '';
            });
            var thetaEl = document.getElementById(p + '-theta-bar');
            if (thetaEl) thetaEl.value = '';
        });

        // Clear result box
        var resultBox = document.getElementById('result-box');
        if (resultBox) resultBox.innerHTML = '<p>No result yet.</p>';

        const type = (data.type || '').toUpperCase();
        if (!['LP', 'QP', 'SDP'].includes(type)) {
            alert('JSON must include a "type" field with value "LP", "QP", or "SDP".');
            return;
        }

        // Switch to the correct tab
        const tabId = type.toLowerCase() + '-tab';
        document.getElementById(tabId).click();

        // Set mode toggle (symbolic or numeric)
        const isNumeric = (data.mode || 'symbolic').toLowerCase() === 'numeric';
        if (isNumeric) {
            document.getElementById('mode-numeric-radio').checked = true;
            currentMode = 'numeric';
        } else {
            document.getElementById('mode-symbolic-radio').checked = true;
            currentMode = 'symbolic';
        }
        onModeChange();

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

        // ── Validation helpers ──
        const errors = [];
        const nRows = m => (Array.isArray(m) ? m.length : 0);
        const nCols = m => (Array.isArray(m) && Array.isArray(m[0]) ? m[0].length : 0);
        const isSquare = m => nRows(m) > 0 && nRows(m) === nCols(m);
        const dictSize = d => (d ? Object.keys(d).length : 0);

        if (type === 'LP' || type === 'QP') {
            // ── Infer dimensions ──
            const d = data.c ? nRows(data.c) : (data.n_x || 0);
            const m = data.A_d ? nRows(data.A_d) : (data.rows_A || 0);
            const n = data.G ? nRows(data.G) : 0;

            // ── Validate ──
            if (!data.c || d === 0) errors.push('c (cost vector) is missing or empty.');
            if (isNumeric) {
                if (d === 0) errors.push('n_x (decision variable count) is missing or zero.');
                if (m === 0) errors.push('rows_A (soft constraint rows) is missing or zero.');
            } else {
                if (!data.A_d || m === 0) errors.push('A_d (soft constraint matrix) is missing or empty.');
                if (!data.b_d || nRows(data.b_d) === 0) errors.push('b_d (soft constraint vector) is missing or empty.');

                if (data.A_d && data.c && nCols(data.A_d) !== d)
                    errors.push(`A_d has ${nCols(data.A_d)} columns but c has ${d} rows (d=${d}). Columns of A_d must equal d.`);
                if (data.A_d && data.b_d && nRows(data.A_d) !== nRows(data.b_d))
                    errors.push(`A_d has ${nRows(data.A_d)} rows but b_d has ${nRows(data.b_d)} rows. They must match (m).`);
                if (data.b_d && nCols(data.b_d) !== 1)
                    errors.push(`b_d should have 1 column but has ${nCols(data.b_d)}.`);
            }
            if (data.c && nCols(data.c) !== 1)
                errors.push(`c should have 1 column but has ${nCols(data.c)}.`);

            if (data.G && data.h) {
                if (nRows(data.G) !== nRows(data.h))
                    errors.push(`G has ${nRows(data.G)} rows but h has ${nRows(data.h)} rows. They must match (n).`);
                if (nCols(data.G) !== d)
                    errors.push(`G has ${nCols(data.G)} columns but d=${d}. Columns of G must equal d.`);
                if (nCols(data.h) !== 1)
                    errors.push(`h should have 1 column but has ${nCols(data.h)}.`);
            } else if (data.G && !data.h) {
                errors.push('G is provided but h is missing.');
            } else if (!data.G && data.h) {
                errors.push('h is provided but G is missing.');
            }

            if (type === 'QP' && data.Q) {
                if (!isSquare(data.Q))
                    errors.push(`Q must be square but is ${nRows(data.Q)}x${nCols(data.Q)}.`);
                if (nRows(data.Q) !== d)
                    errors.push(`Q is ${nRows(data.Q)}x${nRows(data.Q)} but d=${d}. Q must be dxd.`);
            }

            if (errors.length > 0) {
                alert('Problem validation errors:\n\n' + errors.join('\n'));
                return;
            }

            // ── Set form values ──
            if (!isNumeric) {
                if (data.A_d) document.getElementById(prefix + '-A_d').value = JSON.stringify(data.A_d);
                if (data.b_d) document.getElementById(prefix + '-b_d').value = JSON.stringify(data.b_d);
            }
            if (data.G) document.getElementById(prefix + '-A').value = JSON.stringify(data.G);
            if (data.h) document.getElementById(prefix + '-b').value = JSON.stringify(data.h);
            if (data.c) document.getElementById(prefix + '-c').value = JSON.stringify(data.c);
            if (type === 'QP' && data.Q) {
                document.getElementById('qp-Q').value = JSON.stringify(data.Q);
            }

            // ── Set dimension controls ──
            document.getElementById('dim-nx').value = d;
            document.getElementById('dim-rows-a').value = m;
            document.getElementById('dim-rows-g').value = n;

        } else if (type === 'SDP') {
            // ── Infer dimensions ──
            const d = data.c ? nRows(data.c) : (data.n_x || 0);
            const lmiSize = (data.F_d && data.F_d["0"]) ? nRows(data.F_d["0"]) : (data.lmi_size || 0);
            const lmiESize = (data.E && data.E["0"]) ? nRows(data.E["0"]) : 0;

            // ── Validate ──
            if (!data.c || d === 0) errors.push('c (cost vector) is missing or empty.');

            if (data.c && nCols(data.c) !== 1)
                errors.push(`c should have 1 column but has ${nCols(data.c)}.`);

            if (isNumeric) {
                if (d === 0) errors.push('n_x (decision variable count) is missing or zero.');
                if (lmiSize === 0) errors.push('lmi_size (LMI matrix size) is missing or zero.');
            } else {
                if (!data.F_d) errors.push('F_d (soft LMI matrices) is missing.');
                if (data.F_d) {
                    const expectedCount = d + 1;
                    const actualCount = dictSize(data.F_d);
                    if (actualCount !== expectedCount)
                        errors.push(`F_d should have ${expectedCount} matrices (F_0..F_d) but has ${actualCount}.`);
                    for (const [key, mat] of Object.entries(data.F_d)) {
                        if (!isSquare(mat))
                            errors.push(`F_d["${key}"] must be square but is ${nRows(mat)}x${nCols(mat)}.`);
                        else if (nRows(mat) !== lmiSize)
                            errors.push(`F_d["${key}"] is ${nRows(mat)}x${nRows(mat)} but F_d["0"] is ${lmiSize}x${lmiSize}. All must be the same size.`);
                    }
                }
            }

            if (data.E) {
                const expectedCount = d + 1;
                const actualCount = dictSize(data.E);
                if (actualCount !== expectedCount)
                    errors.push(`E should have ${expectedCount} matrices (E_0..E_d) but has ${actualCount}.`);
                for (const [key, mat] of Object.entries(data.E)) {
                    if (!isSquare(mat))
                        errors.push(`E["${key}"] must be square but is ${nRows(mat)}x${nCols(mat)}.`);
                    else if (nRows(mat) !== lmiESize)
                        errors.push(`E["${key}"] is ${nRows(mat)}x${nRows(mat)} but E["0"] is ${lmiESize}x${lmiESize}. All must be the same size.`);
                }
            }

            if (data.Q) {
                if (!isSquare(data.Q))
                    errors.push(`Q must be square but is ${nRows(data.Q)}x${nCols(data.Q)}.`);
                if (nRows(data.Q) !== d)
                    errors.push(`Q is ${nRows(data.Q)}x${nRows(data.Q)} but d=${d}. Q must be dxd.`);
            }

            if (errors.length > 0) {
                alert('Problem validation errors:\n\n' + errors.join('\n'));
                return;
            }

            // ── Set form values ──
            if (!isNumeric && data.F_d) {
                document.getElementById('sdp-F_d').value = JSON.stringify(data.F_d);
                // Populate SDP collection so individual buttons show data
                window.sdpMatrixCollection1 = {};
                for (const [key, mat] of Object.entries(data.F_d)) {
                    window.sdpMatrixCollection1[parseInt(key)] = mat;
                }
            }
            if (data.E) {
                document.getElementById('sdp-F').value = JSON.stringify(data.E);
                window.sdpMatrixCollection2 = {};
                for (const [key, mat] of Object.entries(data.E)) {
                    window.sdpMatrixCollection2[parseInt(key)] = mat;
                }
            }
            if (data.c) document.getElementById('sdp-c').value = JSON.stringify(data.c);
            if (data.Q) document.getElementById('sdp-Q').value = JSON.stringify(data.Q);

            // ── Set dimension controls ──
            document.getElementById('dim-nx').value = d;
            document.getElementById('dim-lmi-size').value = lmiSize;
            document.getElementById('dim-lmi-e-size').value = lmiESize || 0;
        }

        // Update hard constraint button state based on new dimensions
        updateHardConstraintState();

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

/**
 * Move #mode-toggle-row and #dimension-controls into the active tab's card.
 * Called on page load and after each tab transition completes.
 */
function relocateSharedElements() {
    var activePane = document.querySelector('.tab-pane.active');
    if (!activePane) return;

    var popoverOpts = { trigger: 'hover', html: true };

    // Move mode toggle into the active tab's title row
    var modeToggle = document.getElementById('mode-toggle-row');
    var placeholder = activePane.querySelector('.mode-toggle-placeholder');
    if (modeToggle && placeholder) {
        $(modeToggle).find('[data-toggle="popover"]').popover('dispose');
        placeholder.appendChild(modeToggle);
        $(modeToggle).find('[data-toggle="popover"]').popover(popoverOpts);
    }

    // Move dimension controls below the LaTeX div
    var dimControls = document.getElementById('dimension-controls');
    var activeTab = document.querySelector('.nav-link.active');
    if (!dimControls || !activeTab) return;
    var tabId = activeTab.id.replace('-tab', '');
    var latexDiv = document.getElementById(tabId + '-latex');
    if (latexDiv) {
        $(dimControls).find('[data-toggle="popover"]').popover('dispose');
        latexDiv.insertAdjacentElement('afterend', dimControls);
        $(dimControls).find('[data-toggle="popover"]').popover(popoverOpts);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize mode toggle
    document.querySelectorAll('input[name="mode"]').forEach(function(radio) {
        radio.addEventListener('change', function() {
            currentMode = this.value;
            onModeChange();
        });
    });

    // Set initial mode help content, then move shared elements into active tab
    updateModeHelpContent();
    relocateSharedElements();

    // Initialize popovers for static help icons (not relocated by JS)
    $('.dim-help-btn').not('#mode-toggle-row .dim-help-btn, #dimension-controls .dim-help-btn').popover({
        trigger: 'hover',
        html: true
    });

    // Render LaTeX inside popovers when they appear
    $(document).on('shown.bs.popover', function() {
        MathJax.typesetPromise();
    });

    // Filter solvers and update dimension visibility on tab change
    $('a[data-toggle="tab"]').on('shown.bs.tab', function() {
        updateModeHelpContent();
        relocateSharedElements();
        filterSolvers();
        updateDimensionVisibility();
        updateConfidenceLabel();
        updateScenarioLabel();
    });

    // Initial filter on page load
    filterSolvers();
    updateDimensionVisibility();
    updateConfidenceLabel();
    updateScenarioLabel();

    // Update sweep counter when rho/tau fields change
    ['lp', 'qp', 'sdp'].forEach(function(tab) {
        var rhoEl = document.getElementById(tab + '-rho');
        var tauEl = document.getElementById(tab + '-tau');
        if (rhoEl) rhoEl.addEventListener('input', function() { updateSweepCounter(); });
        if (tauEl) tauEl.addEventListener('input', function() { updateSweepCounter(); });
    });

    // Update hard constraint button state when dimension inputs change
    document.getElementById('dim-rows-g').addEventListener('input', updateHardConstraintState);
    document.getElementById('dim-lmi-e-size').addEventListener('input', updateHardConstraintState);

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