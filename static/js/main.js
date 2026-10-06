window.sdpMatrixCollection1 = {};
window.sdpMatrixCollection2 = {};
window.sdpMatrixCollection3 = {};
window.sdpMatrixCollection4 = {};
window.sdpCurrentMatrixIndex = null;
let currentMatrix = '';
let lastResultData = null;
let currentMode = 'symbolic';

/**
 * Safe wrappers around MathJax — MathJax is loaded async from a CDN and
 * may not be defined when our code fires (esp. during DOMContentLoaded).
 * Any bare reference throws ReferenceError, which aborted the rest of
 * init and silently broke the MOSEK license listeners. These helpers
 * never throw: if MathJax isn't ready they're a no-op.
 */
function safeTypeset(arg) {
    try {
        if (typeof MathJax !== 'undefined' && MathJax.typeset) {
            if (arg === undefined) MathJax.typeset();
            else MathJax.typeset(arg);
        }
    } catch (e) { /* MathJax hiccup — don't take the page down */ }
}
function safeTypesetPromise(arg) {
    try {
        if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
            return arg === undefined ? MathJax.typesetPromise() : MathJax.typesetPromise(arg);
        }
    } catch (e) { /* ignore */ }
    return Promise.resolve();
}

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
        } else if (currentMatrix === 'sdp-E') {
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
    safeTypesetPromise();
}

function openSDPModal(tab, matrix) {
    currentMatrix = `${tab}-${matrix}`;
    const matrixInput = document.getElementById(currentMatrix);
    let matrixValues;

    // Create dynamic buttons
    updateSDPButtons();

    // Clear SDP JSON paste box so a stale prior paste doesn't override individual entries.
    const sdpPasteBox = document.getElementById('sdp-paste');
    if (sdpPasteBox) sdpPasteBox.value = '';
    clearModalErrors();
    setSDPPasteHints(matrix);

    // Set default dimensions based on the matrix
    var titleMap = {
        'F_d': `Edit \\(F_j(\\delta)\\)`, 'E': `Edit \\(E_j\\)`,
        'A_da': `Edit \\(A_j(\\delta)\\)`, 'A_a': `Edit \\(G_j\\)`,
        'b_da': `Edit \\(b_j(\\delta)\\)`, 'b_a': `Edit \\(h_j\\)`
    };
    if (titleMap[matrix]) {
        matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : [[0, 0]];
        updateMatrixGrid(matrixValues);
        document.getElementById('matrixModalTitle').textContent = `${titleMap[matrix]} (${tab.toUpperCase()})`;
        document.getElementById('sdpModalTip').innerHTML = `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
        updateSDPFormatReference(matrix);
        safeTypesetPromise();
        $('#SDPModal').modal('show');
        if (typeof validateSolveInputs === 'function') validateSolveInputs();
    }
}

function openSDPMatrixEditor(idx) {

    var collectionMap = {
        'F_d': 'sdpMatrixCollection1', 'A_da': 'sdpMatrixCollection1',
        'E': 'sdpMatrixCollection2', 'A_a': 'sdpMatrixCollection2',
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
        var existing = window[collName][idx] ||
            Array.from({length: sdpCols}, function() { return Array(sdpCols).fill(0); });
        // Always reshape to current LMI size so dim changes are reflected
        var values = reshapeMatrix(existing, sdpCols, sdpCols);
        updateMatrixGrid(values);
        document.getElementById('matrixModalTitle').textContent = 'Edit Matrix ' + idx;
        // Reset paste/error state and show bespoke paste hints for this F_j/E_j.
        var mpaste = document.getElementById('matrix-paste');
        if (mpaste) mpaste.value = '';
        clearModalErrors();
        setMatrixPasteHints(suffix);
        $('#matrixModal').modal('show');
        if (typeof validateSolveInputs === 'function') validateSolveInputs();
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
        case 'G':       return {rows: Math.max(rowsG, 1), cols: nx, colDisabled: false, vals: zeros(Math.max(rowsG, 1), nx)};
        case 'h':       return {rows: Math.max(rowsG, 1), cols: 1,  colDisabled: true,  vals: zeros(Math.max(rowsG, 1), 1)};
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

    let matrixValues = matrixInput.value ? JSON.parse(matrixInput.value) : def.vals;
    // Reshape stored values to current dimension controls: pad with 0s or truncate
    matrixValues = reshapeMatrix(matrixValues, def.rows, def.cols);
    updateMatrixGrid(matrixValues);

    // Clear JSON paste box so a stale prior paste doesn't override the grid.
    const pasteBox = document.getElementById('matrix-paste');
    if (pasteBox) pasteBox.value = '';
    clearModalErrors();
    setMatrixPasteHints(matrix);

    const titles = {
        'A_d': `Edit A(\\(\\delta)\\)`, 'b_d': `Edit b(\\(\\delta)\\)`,
        'c': 'Edit c', 'Q': 'Edit Q', 'G': 'Edit G', 'h': 'Edit h',
        'C': 'Edit C', 'theta-bar': `Edit \\(\\bar{x}\\)`,
        'Theta-bar': `Edit \\(\\bar{X}\\)`
    };
    document.getElementById('matrixModalTitle').textContent =
        `${titles[matrix] || 'Edit ' + matrix} (${tab.toUpperCase()})`;
    document.getElementById('matrixModalTip').innerHTML =
        `<u>Top tip:</u> ${getDynamicTip(matrix)}`;
    updateFormatReference(matrix);
    safeTypesetPromise();
    $('#matrixModal').modal('show');
    if (typeof validateSolveInputs === 'function') validateSolveInputs();
}

function getDynamicTip(matrix) {
    if (matrix === 'A_d') {
        return "Manually enter the matrix. Cells accept expressions with <code>delta[k]</code> (delta[0] = 1st component of vector \\(\\delta\\), delta[1] = 2nd component of vector \\(\\delta\\), ...) including +, -, *, /, ** (or ^) and functions such as <code>sin()</code>, <code>exp()</code>, <code>sqrt()</code>.";
    } else if (matrix === 'b_d') {
        return "Manually enter the vector. Cells accept expressions with <code>delta[k]</code> (delta[0] = 1st component of vector \\(\\delta\\), delta[1] = 2nd component of vector \\(\\delta\\), ...) including +, -, *, /, ** (or ^) and functions such as <code>sin()</code>, <code>exp()</code>, <code>sqrt()</code>.";
    } else if (matrix === 'Q') {
        return "Manually enter the matrix. \\(Q\\) must be <b>symmetric</b> and <b>positive semidefinite</b> (\\(Q \\succeq 0\\)).";
    } else if (matrix === 'c') {
        return "Manually enter the cost vector.";
    } else if (matrix === 'G') {
        return "Manually enter the hard constraint matrix.";
    } else if (matrix === 'h') {
        return "Manually enter the hard constraint vector.";
    } else if (matrix === 'F_d') {
        return "Manually enter the matrices. Each \\(F_j(\\delta)\\) must be <b>symmetric</b>. Cells accept expressions with <code>delta[k]</code> (delta[0] = 1st component of vector \\(\\delta\\), delta[1] = 2nd component of vector \\(\\delta\\), ...) including +, -, *, /, ** (or ^) and functions such as <code>sin()</code>, <code>exp()</code>, <code>sqrt()</code>.";
    } else if (matrix === 'E') {
        return "Manually enter the matrices. Each \\(E_j\\) must be <b>symmetric</b>.";
    } else if (matrix === 'A_da') {
        return "Manually enter the matrices. Each \\(A_j(\\delta)\\) must be <b>symmetric</b>. Cells accept expressions with <code>delta[k]</code> (delta[0] = 1st component of vector \\(\\delta\\), delta[1] = 2nd component of vector \\(\\delta\\), ...) including +, -, *, /, ** (or ^) and functions such as <code>sin()</code>, <code>exp()</code>, <code>sqrt()</code>.";
    } else if (matrix === 'A_a') {
        return "Manually enter the matrices. Each \\(G_j\\) must be <b>symmetric</b>.";
    } else if (matrix === 'C') {
        return "Manually enter the matrix. \\(C\\) must be <b>symmetric</b> and <b>positive semidefinite</b>.";
    } else if (matrix === 'Theta-bar') {
        return "Manually enter the reference matrix.";
    } else if (matrix === 'theta-bar') {
        return "Manually enter the reference vector.";
    } else if (matrix === 'b_da') {
        return "Manually enter the vectors. Cells accept expressions with <code>delta[k]</code> (delta[0] = 1st component of vector \\(\\delta\\), delta[1] = 2nd component of vector \\(\\delta\\), ...) including +, -, *, /, ** (or ^) and functions such as <code>sin()</code>, <code>exp()</code>, <code>sqrt()</code>.";
    } else if (matrix === 'b_a') {
        return "Manually enter the vectors.";
    }
    return "Manually enter the data.";
}

/* ── Bespoke Input Format Reference for the matrix modal ── */
function updateFormatReference(matrix) {
    const ref = document.getElementById('matrixFormatRef');
    if (!ref) return;

    const pre = 'style="background:#f8f9fa; padding:6px; border-radius:4px; font-size:0.85em; margin:4px 0;"';

    // Category — delta-capable matrices only exist in Symbolic mode.
    const isDelta = ['A_d', 'b_d'].includes(matrix);
    const isQ = (matrix === 'Q');
    const isRef = (matrix === 'theta-bar' || matrix === 'Theta-bar');

    const dims = (function () {
        switch (matrix) {
            case 'c':         return '\\(d \\times 1\\) (column vector of length \\(d\\))';
            case 'Q':         return '\\(d \\times d\\) (symmetric, \\(Q \\succeq 0\\))';
            case 'A_d':       return '\\(\\mathfrak{m} \\times d\\) (one row per scenario constraint)';
            case 'b_d':       return '\\(\\mathfrak{m} \\times 1\\) (column vector)';
            case 'G':         return '\\(\\mathfrak{n} \\times d\\) (hard constraint coefficients)';
            case 'h':         return '\\(\\mathfrak{n} \\times 1\\) (hard constraint right-hand side)';
            case 'theta-bar': return '\\(d \\times 1\\) (reference vector \\(\\bar{x}\\) for regularization)';
            case 'Theta-bar': return '\\(d \\times d\\)';
            default:          return 'auto-sized from the dimension controls';
        }
    })();

    // ── Cell content rules ──
    let html = '<h6><b>Cell content:</b></h6>';
    if (isDelta) {
        html += '<p>This matrix is <b>delta-capable</b> — cells accept <b>numeric values</b> <i>or</i> expressions that reference ' +
                '<code>delta[k]</code> (<code>delta[0]</code> = 1st component of \\(\\delta\\), <code>delta[1]</code> = 2nd component of \\(\\delta\\), \\(\\ldots\\)). ' +
                'Operators <code>+ - * / **</code> and functions such as <code>sin()</code>, <code>cos()</code>, <code>exp()</code>, <code>log()</code>, <code>sqrt()</code>, <code>abs()</code> and the constants <code>pi</code>, <code>e</code> are allowed (<code>^</code> also means power). ' +
                'Example cells: <code>delta[0] + 1</code>, <code>sin(delta[1])</code>, <code>2.5</code>.</p>';
        html += '<p><b>Mode note:</b> this matrix is only editable when the <b>Symbolic</b> input mode is selected. ' +
                'In <b>Numeric</b> mode, \\(A(\\delta_i)\\) and \\(b(\\delta_i)\\) come directly from the uploaded scenario data ' +
                '(each row = row-wise flattened \\(A_i\\) concatenated with \\(b_i\\)), so this editor is hidden.</p>';
    } else if (isQ) {
        html += '<p>This matrix is <b>numeric-only</b>: cells accept only numeric values. ' +
                '\\(Q\\) must additionally be <b>symmetric</b> and <b>positive semidefinite</b> (\\(Q \\succeq 0\\)). ' +
                '<code>delta[k]</code> expressions are <b>not</b> permitted here.</p>';
    } else if (isRef) {
        html += '<p>This vector is <b>numeric-only</b>: cells accept only numeric values. ' +
                'It is the reference point for the regularization term \\(\\tau\\,\\lVert x - \\bar{x}\\rVert_p\\). ' +
                '<code>delta[k]</code> expressions are <b>not</b> permitted here.</p>';
    } else {
        html += '<p>This matrix is <b>numeric-only</b>: cells accept only numeric values. ' +
                '<code>delta[k]</code> expressions are <b>not</b> permitted here.</p>';
    }

    html += `<h6><b>Expected shape:</b></h6><p>${dims}. The grid is auto-sized from the dimension controls (\\(d\\), \\(\\mathfrak{m}\\), \\(\\mathfrak{n}\\)).</p>`;

    // ── JSON paste format ──
    html += '<h6><b>JSON paste format:</b></h6>';
    html += `<p>A 2-D JSON array whose outer length is the number of rows and whose inner arrays are the columns. ` +
            `Each cell is ${isDelta
                ? 'either a <b>number</b> (e.g. <code>1.5</code>) or a <b>string</b> containing a <code>delta[k]</code> expression (e.g. <code>"delta[0]+1"</code>)'
                : 'a <b>number</b> (e.g. <code>1.5</code>) — <b>no strings, no <code>delta[k]</code></b>'}. ` +
            `If a non-empty paste is present on Save, it replaces the grid contents.</p>`;
    if (isDelta) {
        html += `<pre ${pre}>[
  ["delta[0] + 1",  0,              "sin(delta[1])"],
  [0,               "2 * delta[0]", -1.5]
]</pre>`;
    } else if (isQ) {
        html += `<pre ${pre}>[
  [2.0, 0.5],
  [0.5, 3.0]
]</pre>`;
    } else if (matrix === 'c' || matrix === 'h' || isRef) {
        html += `<pre ${pre}>[
  [1.0],
  [-2.0]
]</pre>`;
    } else {
        html += `<pre ${pre}>[
  [1.0, 0.0],
  [0.0, -1.5]
]</pre>`;
    }

    html += '<p class="text-muted"><small>For file-based matrix input (CSV / MAT / NPY / …), use the <b>Upload Program</b> button at the top of the page.</small></p>';

    ref.innerHTML = html;
    safeTypesetPromise([ref]);
}

/* ── Bespoke Input Format Reference for the SDP collection modal ── */
function updateSDPFormatReference(matrix) {
    const ref = document.getElementById('sdpFormatRef');
    if (!ref) return;

    const pre = 'style="background:#f8f9fa; padding:6px; border-radius:4px; font-size:0.85em; margin:4px 0;"';
    const isSoft = ['F_d', 'A_da', 'b_da'].includes(matrix);  // delta-capable collections

    // Symbol names for descriptions
    const symbolName = isSoft ? '\\(F_j(\\delta)\\)' : '\\(E_j\\)';
    const sizeSym = isSoft ? '\\(\\mathfrak{m} \\times \\mathfrak{m}\\)' : '\\(\\mathfrak{n} \\times \\mathfrak{n}\\)';

    let html = '<h6><b>What is being edited:</b></h6>';
    html += `<p>A <b>collection</b> of matrices <code>${symbolName.replace(/\\\\/g, '\\')}</code> for ` +
            '\\(j = 0, 1, \\ldots, d\\) — i.e. <b>\\(d+1\\) square matrices</b> of size ' + sizeSym +
            ` that together define the LMI constraint ${isSoft
                ? '\\(F_0(\\delta_i) + \\sum_{j=1}^{d} x_j F_j(\\delta_i) \\preceq 0\\)'
                : '\\(E_0 + \\sum_{j=1}^{d} x_j E_j \\preceq 0\\)'}.</p>`;

    html += '<h6><b>Cell content:</b></h6>';
    if (isSoft) {
        html += '<p>Each matrix is <b>delta-capable</b> — cells accept <b>numeric values</b> <i>or</i> expressions that reference ' +
                '<code>delta[k]</code> (<code>delta[0]</code> = 1st component of \\(\\delta\\), <code>delta[1]</code> = 2nd component of \\(\\delta\\), \\(\\ldots\\)). ' +
                'Operators <code>+ - * / **</code> and functions such as <code>sin()</code>, <code>cos()</code>, <code>exp()</code>, <code>log()</code>, <code>sqrt()</code>, <code>abs()</code> and the constants <code>pi</code>, <code>e</code> are allowed (<code>^</code> also means power). ' +
                'Every matrix must be <b>symmetric</b>.</p>';
        html += '<p><b>Mode note:</b> scenario LMI matrices \\(F_j(\\delta)\\) are only editable when the <b>Symbolic</b> input mode is selected. ' +
                'In <b>Numeric</b> mode, \\(F_{0,i}, F_{1,i}, \\ldots, F_{d,i}\\) come directly from the uploaded scenario data ' +
                '(each row = row-wise flattened \\(F_{0,i}\\) then \\(F_{1,i}\\) then \\(\\ldots\\) then \\(F_{d,i}\\) concatenated), so this editor is hidden.</p>';
    } else {
        html += '<p>Each matrix is <b>numeric-only</b>: cells accept only numeric values. ' +
                'Every matrix must be <b>symmetric</b>. <code>delta[k]</code> expressions are <b>not</b> permitted here.</p>';
    }

    html += `<h6><b>Expected shape:</b></h6><p>Exactly <b>\\(d+1\\)</b> matrices (indexed <code>"0"</code> through <code>"d"</code>), each ${sizeSym}. ` +
            'The per-matrix editor grid is auto-sized from the LMI dimension control.</p>';

    // ── JSON paste format ──
    html += '<h6><b>JSON paste format (full collection):</b></h6>';
    html += `<p>A <b>dictionary / JSON object</b> keyed by stringified matrix index — <code>"0"</code>, <code>"1"</code>, \\(\\ldots\\), <code>"d"</code>. ` +
            `Each value is a 2-D JSON array where each cell is ${isSoft
                ? 'either a <b>number</b> or a <b>string</b> containing a <code>delta[k]</code> expression'
                : 'a <b>number</b> — <b>no strings, no <code>delta[k]</code></b>'}. ` +
            'If a non-empty paste is present on Save, it <b>replaces the entire collection</b> (overriding any per-matrix entries made via the buttons above).</p>';
    if (isSoft) {
        html += `<pre ${pre}>{
  "0": [["delta[0]",       0], [0,               1]],
  "1": [[1,                0], [0,              -1]],
  "2": [[0,     "delta[1]"], ["delta[1]",       0]]
}</pre>`;
    } else {
        html += `<pre ${pre}>{
  "0": [[1, 0], [0,  1]],
  "1": [[0, 1], [1,  0]],
  "2": [[-1, 0], [0, -1]]
}</pre>`;
    }

    html += '<p class="text-muted"><small>For file-based collection input (MAT / JSON-file), use the <b>Upload Program</b> button at the top of the page.</small></p>';

    ref.innerHTML = html;
    safeTypesetPromise([ref]);
}

/* ── Cell validation for matrix modals ─────────────────────────────────── */

/**
 * Is `currentMatrix` one that is allowed to contain delta[k] expressions?
 * Delta-capable kinds: A(δ), b(δ), and any SDP scenario collection (F_d / A_da / b_da).
 * Everything else (c, Q, G=A, h=b, x̄, E_j=F, etc.) is numeric-only.
 */
function isCurrentMatrixDeltaCapable() {
    if (!currentMatrix) return false;
    // currentMatrix format is "<tab>-<kind>" (e.g. "lp-A_d", "sdp-F_d").
    const kind = currentMatrix.split('-').slice(1).join('-');
    return ['A_d', 'b_d', 'F_d', 'A_da', 'b_da'].includes(kind);
}

/**
 * Validate a single cell's text against the delta-capable rule:
 *   • numeric-only → must parse as a finite number
 *   • delta-capable → either a finite number, or an expression in delta[k]
 * Returns '' if valid, otherwise a short reason.
 */
function validateCellText(text, deltaCapable) {
    const s = String(text == null ? '' : text).trim();
    if (s === '') return 'Empty cell.';
    // Accept things parseable as a number
    const n = Number(s);
    if (Number.isFinite(n)) return '';
    if (deltaCapable) {
        // Allow any non-numeric text — parser will validate on solve.
        // Flag only blatantly empty / nonsense content.
        return '';
    }
    // Numeric-only path: reject delta[...] and any non-numeric string.
    if (/delta\s*\[/i.test(s)) {
        return 'delta[k] is not allowed in a numeric-only matrix.';
    }
    return 'Cell must be a number (e.g. 1.5, -2, 0).';
}

/**
 * Apply validation to every <input> in the matrix grid: toggle red outline,
 * collect errors, and surface a summary message. Returns {valid, message}.
 */
function validateMatrixGrid() {
    const grid = document.getElementById('matrixGrid');
    const errBox = document.getElementById('matrixModalError');
    if (!grid) return {valid: true, message: ''};
    const deltaCapable = isCurrentMatrixDeltaCapable();
    const inputs = Array.from(grid.querySelectorAll('input'));
    let firstReason = '';
    let badCount = 0;
    inputs.forEach(inp => {
        const reason = validateCellText(inp.value, deltaCapable);
        if (reason) {
            inp.classList.add('cell-invalid');
            badCount++;
            if (!firstReason) firstReason = reason;
        } else {
            inp.classList.remove('cell-invalid');
        }
    });
    if (errBox) {
        if (badCount > 0) {
            errBox.textContent = `Invalid entries (${badCount}): ${firstReason}`;
            errBox.style.display = 'block';
        } else {
            errBox.textContent = '';
            errBox.style.display = 'none';
        }
    }
    return {valid: badCount === 0, message: firstReason};
}

/**
 * Wire up live validation on the matrix grid. Called after updateMatrixGrid().
 */
function attachGridValidationListeners() {
    const grid = document.getElementById('matrixGrid');
    if (!grid) return;
    grid.querySelectorAll('input').forEach(inp => {
        // Avoid double-binding
        if (inp.dataset.validationBound === '1') return;
        inp.dataset.validationBound = '1';
        ['input', 'blur'].forEach(evt => inp.addEventListener(evt, validateMatrixGrid));
    });
    // Run initial pass so the state reflects whatever is loaded into the grid.
    validateMatrixGrid();
}

/**
 * Set bespoke placeholder + caption for the per-matrix JSON paste box, tuned
 * to the specific matrix kind being edited.
 */
function setMatrixPasteHints(kind) {
    const ta = document.getElementById('matrix-paste');
    const cap = document.getElementById('matrixPasteCaption');
    if (!ta || !cap) return;
    const hints = {
        'c':          {
            placeholder: '[[1], [-2]]',
            caption: 'Cost vector \\(c\\) — a \\(d \\times 1\\) 2-D JSON array of <b>numbers only</b>. Loads into the grid above on Save.'
        },
        'Q':          {
            placeholder: '[[2, 0.5], [0.5, 3]]',
            caption: 'Quadratic cost matrix \\(Q\\) — a \\(d \\times d\\) 2-D JSON array of <b>numbers only</b>; must be symmetric and \\(\\succeq 0\\). Loads into the grid above on Save.'
        },
        'A_d':        {
            placeholder: '[["-delta[0]", 0], [0, "-delta[0]"]]',
            caption: 'Scenario constraint matrix \\(A(\\delta)\\) — a \\(\\mathfrak{m} \\times d\\) 2-D JSON array of <b>numbers</b> or <b>strings containing <code>delta[k]</code></b> expressions. Loads into the grid above on Save.'
        },
        'b_d':        {
            placeholder: '[["delta[0] - 1"], ["delta[0] - 1"]]',
            caption: 'Scenario constraint vector \\(b(\\delta)\\) — a \\(\\mathfrak{m} \\times 1\\) 2-D JSON array of <b>numbers</b> or <b>strings containing <code>delta[k]</code></b> expressions. Loads into the grid above on Save.'
        },
        'G':          {
            placeholder: '[[-1, 0], [0, -1]]',
            caption: 'Hard constraint matrix \\(G\\) — a \\(\\mathfrak{n} \\times d\\) 2-D JSON array of <b>numbers only</b>. Loads into the grid above on Save.'
        },
        'h':          {
            placeholder: '[[0], [0]]',
            caption: 'Hard constraint vector \\(h\\) — a \\(\\mathfrak{n} \\times 1\\) 2-D JSON array of <b>numbers only</b>. Loads into the grid above on Save.'
        },
        'theta-bar':  {
            placeholder: '[[0], [0]]',
            caption: 'Regularization reference point \\(\\bar{x}\\) — a \\(d \\times 1\\) 2-D JSON array of <b>numbers only</b>. Loads into the grid above on Save.'
        }
    };
    // Per-F_j / E_j editing opens the matrix modal too, via the SDP flow.
    // We detect that via window.sdpCurrentMatrixIndex being non-null.
    if (window.sdpCurrentMatrixIndex !== null) {
        const isHard = ['E', 'A_a', 'b_a'].includes(kind);
        const lmiId = isHard ? 'dim-lmi-e-size' : 'dim-lmi-size';
        const sz = parseInt(document.getElementById(lmiId).value) || 2;
        if (isHard) {
            ta.placeholder = '[[1, 0], [0, -1]]';
            cap.innerHTML = `Hard LMI matrix \\(E_{${window.sdpCurrentMatrixIndex}}\\) — a ${sz}×${sz} 2-D JSON array of <b>numbers only</b>; must be symmetric. Loads into the grid above on Save.`;
        } else {
            ta.placeholder = '[["delta[0]", 0], [0, 1]]';
            cap.innerHTML = `Scenario LMI matrix \\(F_{${window.sdpCurrentMatrixIndex}}(\\delta)\\) — a ${sz}×${sz} 2-D JSON array of <b>numbers</b> or <b>strings containing <code>delta[k]</code></b> expressions; must be symmetric. Loads into the grid above on Save.`;
        }
        safeTypesetPromise([cap]);
        return;
    }
    const h = hints[kind] || {
        placeholder: '[[1, 0], [0, 1]]',
        caption: 'Paste a 2-D JSON array. Loads into the grid above on Save.'
    };
    ta.placeholder = h.placeholder;
    cap.innerHTML = h.caption;
    safeTypesetPromise([cap]);
}

/**
 * Set bespoke placeholder + caption for the SDP collection JSON paste box.
 */
function setSDPPasteHints(kind) {
    const ta = document.getElementById('sdp-paste');
    const cap = document.getElementById('sdpPasteCaption');
    if (!ta || !cap) return;
    const nx = parseInt(document.getElementById('dim-nx').value) || 0;
    const isHard = ['E', 'A_a', 'b_a'].includes(kind);
    const sizeId = isHard ? 'dim-lmi-e-size' : 'dim-lmi-size';
    const sz = parseInt(document.getElementById(sizeId).value) || 0;
    const dMax = Math.max(nx, 0);

    if (kind === 'F_d') {
        ta.placeholder = '{\n  "0": [["delta[0]", 0], [0, 1]],\n  "1": [[1, 0], [0, -1]]\n}';
        cap.innerHTML = `Scenario LMI collection \\(F_j(\\delta)\\) for \\(j = 0, 1, \\ldots, d\\) — a dictionary with <b>${dMax + 1}</b> entries keyed <code>"0"</code>…<code>"${dMax}"</code>. Each value is a <b>${sz || 'm'}×${sz || 'm'}</b> 2-D JSON array of <b>numbers</b> or <b>strings containing <code>delta[k]</code></b> expressions; each matrix must be symmetric.`;
    } else if (kind === 'E') {
        ta.placeholder = '{\n  "0": [[1, 0], [0, 1]],\n  "1": [[0, 1], [1, 0]]\n}';
        cap.innerHTML = `Hard LMI collection \\(E_j\\) for \\(j = 0, 1, \\ldots, d\\) — a dictionary with <b>${dMax + 1}</b> entries keyed <code>"0"</code>…<code>"${dMax}"</code>. Each value is a <b>${sz || 'n'}×${sz || 'n'}</b> 2-D JSON array of <b>numbers only</b>; each matrix must be symmetric.`;
    } else if (kind === 'A_da') {
        ta.placeholder = '{\n  "0": [["delta[0]", 0], [0, 1]],\n  "1": [[1, 0], [0, -1]]\n}';
        cap.innerHTML = `Scenario collection \\(A_j(\\delta)\\) for \\(j = 0, 1, \\ldots, d\\) — a dictionary with <b>${dMax + 1}</b> entries keyed <code>"0"</code>…<code>"${dMax}"</code>. Each value is a 2-D JSON array of <b>numbers</b> or <b>strings containing <code>delta[k]</code></b> expressions; each matrix must be symmetric.`;
    } else if (kind === 'A_a') {
        ta.placeholder = '{\n  "0": [[1, 0], [0, 1]],\n  "1": [[0, 1], [1, 0]]\n}';
        cap.innerHTML = `Hard collection \\(G_j\\) for \\(j = 0, 1, \\ldots, d\\) — a dictionary with <b>${dMax + 1}</b> entries keyed <code>"0"</code>…<code>"${dMax}"</code>. Each value is a 2-D JSON array of <b>numbers only</b>; each matrix must be symmetric.`;
    } else if (kind === 'b_da') {
        ta.placeholder = '{\n  "0": [["delta[0]"], [1]],\n  "1": [[1], [-1]]\n}';
        cap.innerHTML = `Scenario vector collection \\(b_j(\\delta)\\) for \\(j = 0, 1, \\ldots, d\\) — a dictionary with <b>${dMax + 1}</b> entries keyed <code>"0"</code>…<code>"${dMax}"</code>. Each value is a 2-D JSON array of <b>numbers</b> or <b>strings containing <code>delta[k]</code></b> expressions.`;
    } else if (kind === 'b_a') {
        ta.placeholder = '{\n  "0": [[1], [0]],\n  "1": [[0], [1]]\n}';
        cap.innerHTML = `Hard vector collection \\(h_j\\) for \\(j = 0, 1, \\ldots, d\\) — a dictionary with <b>${dMax + 1}</b> entries keyed <code>"0"</code>…<code>"${dMax}"</code>. Each value is a 2-D JSON array of <b>numbers only</b>.`;
    } else {
        ta.placeholder = '{"0": [[1, 0], [0, 1]], "1": [[0, 1], [1, 0]]}';
        cap.innerHTML = 'Dictionary keyed by matrix index. Each value is a 2-D JSON array.';
    }
    safeTypesetPromise([cap]);
}

/** Clear any error display in a modal when it opens. */
function clearModalErrors() {
    ['matrixModalError', 'matrixPasteError', 'sdpPasteError'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.textContent = ''; el.style.display = 'none'; }
    });
}

/**
 * Reshape a 2-D array to (rows x cols): truncate rows/columns that overflow
 * and pad missing ones with 0. Keeps existing cell values where possible so
 * that resizing dimensions preserves the user's entered data.
 */
function reshapeMatrix(vals, rows, cols) {
    const out = [];
    for (let i = 0; i < rows; i++) {
        const srcRow = (Array.isArray(vals) && Array.isArray(vals[i])) ? vals[i] : [];
        const row = [];
        for (let j = 0; j < cols; j++) {
            row.push(srcRow[j] !== undefined ? srcRow[j] : 0);
        }
        out.push(row);
    }
    return out;
}

/**
 * Reshape every stored matrix/collection hidden input to match current dim
 * controls. Keeps existing values where possible, pads with 0s. Called when
 * any dim-* input changes so the next modal open reflects the new dims.
 */
function reshapeAllStoredMatrices() {
    const rowsG = parseInt(document.getElementById('dim-rows-g').value) || 0;

    // When n = 0 (LP/QP), the hard constraints G, h are not part of the
    // problem. Clear any previously stored values so they aren't submitted
    // to the server and aren't carried as stale state in the hidden input.
    if (rowsG === 0) {
        ['lp', 'qp'].forEach(function(tab) {
            ['G', 'h'].forEach(function(m) {
                var el = document.getElementById(tab + '-' + m);
                if (el) el.value = '';
            });
        });
    }

    const simpleMatrices = ['c', 'Q', 'A_d', 'b_d', 'G', 'h', 'theta-bar'];
    ['lp', 'qp', 'sdp'].forEach(function(tab) {
        simpleMatrices.forEach(function(m) {
            var el = document.getElementById(tab + '-' + m);
            if (!el || !el.value) return;
            var parsed;
            try { parsed = JSON.parse(el.value); } catch (e) { return; }
            var def = getDimDefaults(m);
            var resized = reshapeMatrix(parsed, def.rows, def.cols);
            el.value = JSON.stringify(resized);
        });
    });

    // SDP matrix collections (F_d, E). Re-key to run 0..d and resize each to lmi_size².
    const nx = parseInt(document.getElementById('dim-nx').value) || 0;
    const lmi = parseInt(document.getElementById('dim-lmi-size').value) || 2;
    const lmiE = parseInt(document.getElementById('dim-lmi-e-size').value) || 0;
    function reshapeCollection(inputId, collectionVar, size) {
        var el = document.getElementById(inputId);
        if (!el || !el.value) return;
        var parsed;
        try { parsed = JSON.parse(el.value); } catch (e) { return; }
        if (typeof parsed !== 'object' || Array.isArray(parsed)) return;
        var newColl = {};
        for (var j = 0; j <= nx; j++) {
            var existing = parsed[String(j)] || parsed[j] || [];
            newColl[j] = reshapeMatrix(existing, size, size);
        }
        window[collectionVar] = newColl;
        el.value = JSON.stringify(newColl);
    }
    reshapeCollection('sdp-F_d', 'sdpMatrixCollection1', lmi);
    if (lmiE > 0) {
        reshapeCollection('sdp-E', 'sdpMatrixCollection2', lmiE);
    } else {
        // n = 0 for SDP → drop the hard LMI collection E so it isn't
        // included as a constraint on solve.
        var sdpEEl = document.getElementById('sdp-E');
        if (sdpEEl) sdpEEl.value = '';
        window.sdpMatrixCollection2 = {};
    }
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
    // Attach live validation (uses currentMatrix to decide numeric vs delta-capable).
    attachGridValidationListeners();
}

function applyCollectionValues(fileValues) {
    if (currentMatrix.endsWith('F_d') || currentMatrix.endsWith('A_da')) {
        window.sdpMatrixCollection1 = fileValues;
        document.getElementById(currentMatrix).value = JSON.stringify(window.sdpMatrixCollection1);
    } else if (currentMatrix.endsWith('E') || currentMatrix.endsWith('A_a')) {
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
    validateSolveInputs();
}

function saveCollection() {
    const pasteBox = document.getElementById('sdp-paste');
    const errBox = document.getElementById('sdpPasteError');
    const showErr = (msg) => {
        if (errBox) { errBox.textContent = msg; errBox.style.display = 'block'; }
        else alert(msg);
    };
    const clearErr = () => {
        if (errBox) { errBox.textContent = ''; errBox.style.display = 'none'; }
    };
    clearErr();

    const pasted = pasteBox && pasteBox.value.trim();
    if (!pasted) {
        applyCollectionValues(undefined);
        return;
    }

    let parsed;
    try { parsed = JSON.parse(pasted); }
    catch (err) { showErr('Invalid JSON: ' + err.message); return; }
    if (typeof parsed !== 'object' || Array.isArray(parsed) || parsed === null) {
        showErr('Pasted JSON must be a dictionary keyed by matrix index ("0", "1", …).');
        return;
    }

    // Expected count and size from dimension controls
    const nx = parseInt(document.getElementById('dim-nx').value) || 0;
    const expectedCount = nx + 1;
    const kind = currentMatrix ? currentMatrix.split('-').slice(1).join('-') : '';
    const isHard = ['E', 'A_a', 'b_a'].includes(kind);
    const sizeId = isHard ? 'dim-lmi-e-size' : 'dim-lmi-size';
    const expectedSize = parseInt(document.getElementById(sizeId).value) || 0;
    const deltaCapable = isCurrentMatrixDeltaCapable();

    // Key-count check
    const keys = Object.keys(parsed);
    if (keys.length !== expectedCount) {
        showErr(`Expected ${expectedCount} matrices (indices "0"…"${nx}") but received ${keys.length}. Adjust the dimension \\(d\\) or the JSON to match.`);
        return;
    }
    // Key-range check
    for (let j = 0; j <= nx; j++) {
        if (!(String(j) in parsed) && !(j in parsed)) {
            showErr(`Missing matrix at key "${j}". Keys must be "0" through "${nx}".`);
            return;
        }
    }

    // Per-matrix shape + cell-content checks
    for (const [key, mat] of Object.entries(parsed)) {
        if (!Array.isArray(mat) || mat.length === 0 || !Array.isArray(mat[0])) {
            showErr(`Entry "${key}" must be a 2-D array (e.g. [[1,0],[0,1]]).`);
            return;
        }
        const r = mat.length;
        const c = mat[0].length;
        // Rectangular?
        for (let i = 0; i < r; i++) {
            if (!Array.isArray(mat[i]) || mat[i].length !== c) {
                showErr(`Entry "${key}" row ${i} has length ${mat[i] ? mat[i].length : 'n/a'}, expected ${c}.`);
                return;
            }
        }
        // Size check
        if (expectedSize > 0 && (r !== expectedSize || c !== expectedSize)) {
            showErr(`Entry "${key}" is ${r}×${c} but expected ${expectedSize}×${expectedSize} from the LMI size control.`);
            return;
        }
        // Cell-content rules
        for (let i = 0; i < r; i++) {
            for (let j = 0; j < c; j++) {
                const reason = validateCellText(mat[i][j], deltaCapable);
                if (reason) {
                    showErr(`Entry "${key}" cell [${i}][${j}] = ${JSON.stringify(mat[i][j])} — ${reason}`);
                    return;
                }
            }
        }
    }

    // Normalise keys to integers and rebuild a clean object
    const clean = {};
    for (const [key, mat] of Object.entries(parsed)) clean[parseInt(key, 10)] = mat;
    applyCollectionValues(clean);
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

function saveMatrix() {
    const grid = document.getElementById('matrixGrid');
    const rowDivs = grid.querySelectorAll('.matrix-row');
    const rows = rowDivs.length;
    const columns = rows > 0 ? rowDivs[0].querySelectorAll('input').length : 0;
    let values = Array.from({length: rows}, () => Array(columns).fill(0));

    // Expected shape for shape validation: prefer dim-default when known; for
    // SDP per-matrix edits (F_j / E_j / etc.) fall back to the grid's shape.
    const kind = currentMatrix ? currentMatrix.split('-').slice(1).join('-') : '';
    let expectedRows = null, expectedCols = null;
    if (window.sdpCurrentMatrixIndex === null) {
        const def = getDimDefaults(kind);
        expectedRows = def.rows;
        expectedCols = def.cols;
    } else {
        // Per-F_j editing in SDP collection: expect square, LMI-size.
        const isHard = ['E', 'A_a', 'b_a'].includes(kind);
        const sizeId = isHard ? 'dim-lmi-e-size' : 'dim-lmi-size';
        const sz = parseInt(document.getElementById(sizeId).value) || 2;
        expectedRows = sz;
        expectedCols = sz;
    }

    const deltaCapable = isCurrentMatrixDeltaCapable();
    const pasteErrBox = document.getElementById('matrixPasteError');
    const gridErrBox = document.getElementById('matrixModalError');

    function showPasteErr(msg) {
        if (pasteErrBox) {
            pasteErrBox.textContent = msg;
            pasteErrBox.style.display = 'block';
        } else {
            alert(msg);
        }
    }
    function showGridErr(msg) {
        if (gridErrBox) {
            gridErrBox.textContent = msg;
            gridErrBox.style.display = 'block';
        } else {
            alert(msg);
        }
    }

    // JSON paste box takes priority over the grid when non-empty.
    const pasteBox = document.getElementById('matrix-paste');
    const pasted = pasteBox && pasteBox.value.trim();
    if (pasted) {
        let parsed;
        try { parsed = JSON.parse(pasted); }
        catch (err) { showPasteErr('Invalid JSON: ' + err.message); return; }

        if (!Array.isArray(parsed) || parsed.length === 0 || !Array.isArray(parsed[0])) {
            showPasteErr('Pasted JSON must be a 2-D array (e.g. [[1,0],[0,1]]).');
            return;
        }
        // Rectangular?
        const rCols = parsed[0].length;
        for (let r = 0; r < parsed.length; r++) {
            if (!Array.isArray(parsed[r]) || parsed[r].length !== rCols) {
                showPasteErr(`Row ${r} has length ${parsed[r] ? parsed[r].length : 'n/a'}, expected ${rCols}. All rows must have the same number of columns.`);
                return;
            }
        }
        // Shape check
        if (expectedRows != null && (parsed.length !== expectedRows || rCols !== expectedCols)) {
            showPasteErr(`Shape mismatch: pasted matrix is ${parsed.length}×${rCols} but expected ${expectedRows}×${expectedCols} from the dimension controls.`);
            return;
        }
        // Cell-content rules
        for (let r = 0; r < parsed.length; r++) {
            for (let c = 0; c < parsed[r].length; c++) {
                const reason = validateCellText(parsed[r][c], deltaCapable);
                if (reason) {
                    showPasteErr(`Cell [${r}][${c}] = ${JSON.stringify(parsed[r][c])} — ${reason}`);
                    return;
                }
            }
        }
        values = parsed;
        // Refresh grid view to match pasted shape
        updateMatrixGrid(values);
    } else {
        // Live-validate the grid before accepting it.
        const check = validateMatrixGrid();
        if (!check.valid) {
            showGridErr(`Fix the highlighted cell(s) before saving. ${check.message}`);
            return;
        }
        // Shape sanity on the grid too (catches out-of-sync state).
        if (expectedRows != null && (rows !== expectedRows || columns !== expectedCols)) {
            showGridErr(`Shape mismatch: grid is ${rows}×${columns} but expected ${expectedRows}×${expectedCols} from the dimension controls.`);
            return;
        }
        Array.from(grid.querySelectorAll('input')).forEach(input => {
            const row = parseInt(input.dataset.row);
            const column = parseInt(input.dataset.column);
            values[row][column] = input.value;
        });
    }

    // If editing from SDP modal, store in collection
    if (window.sdpCurrentMatrixIndex !== null) {
        if (currentMatrix.endsWith('F_d') || currentMatrix.endsWith('A_da')) {
            window.sdpMatrixCollection1[window.sdpCurrentMatrixIndex] = values;
        } else if (currentMatrix.endsWith('E') || currentMatrix.endsWith('A_a')) {
            window.sdpMatrixCollection2[window.sdpCurrentMatrixIndex] = values;
        } else if (currentMatrix.endsWith('b_da')) {
            window.sdpMatrixCollection3[window.sdpCurrentMatrixIndex] = values;
        } else if (currentMatrix.endsWith('b_a')) {
            window.sdpMatrixCollection4[window.sdpCurrentMatrixIndex] = values;
        }
        // Persist collection to the hidden input so server receives latest state
        const prefix = currentMatrix;
        if (prefix.endsWith('F_d') || prefix.endsWith('A_da')) {
            document.getElementById(prefix).value = JSON.stringify(window.sdpMatrixCollection1);
        } else if (prefix.endsWith('E') || prefix.endsWith('A_a')) {
            document.getElementById(prefix).value = JSON.stringify(window.sdpMatrixCollection2);
        } else if (prefix.endsWith('b_da')) {
            document.getElementById(prefix).value = JSON.stringify(window.sdpMatrixCollection3);
        } else if (prefix.endsWith('b_a')) {
            document.getElementById(prefix).value = JSON.stringify(window.sdpMatrixCollection4);
        }
        window.sdpCurrentMatrixIndex = null;
        $('#matrixModal').modal('hide');
        validateSolveInputs();
        return;
    }
    document.getElementById(currentMatrix).value = JSON.stringify(values);
    $('#matrixModal').modal('hide');
    validateSolveInputs();
}

/* ── Pre-solve validation ──────────────────────────────────────────────── */

/**
 * Validate that every stored matrix/collection on the active tab has a shape
 * consistent with the current dimension controls (d, m, n, lmi, lmi_e).
 * Returns {valid: bool, message: string}. `message` is the first mismatch.
 */
function validateSolveInputs() {
    const errEl = document.getElementById('solve-error');
    const setMsg = (msg) => {
        if (!errEl) return;
        if (msg) { errEl.textContent = msg; errEl.style.display = 'block'; }
        else { errEl.textContent = ''; errEl.style.display = 'none'; }
    };

    const activeTab = document.querySelector('.nav-link.active');
    if (!activeTab) { setMsg(''); return {valid: true, message: ''}; }
    const tabId = activeTab.id;
    const tab = tabId.replace('-tab', '');
    const d = parseInt(document.getElementById('dim-nx').value) || 0;
    const m = parseInt(document.getElementById('dim-rows-a').value) || 0;
    const n = parseInt(document.getElementById('dim-rows-g').value) || 0;
    const lmi = parseInt(document.getElementById('dim-lmi-size').value) || 0;
    const lmiE = parseInt(document.getElementById('dim-lmi-e-size').value) || 0;

    // Drive the Edit G / Edit h toggle from the same authoritative n check
    // that this validator uses — so every trigger of validateSolveInputs()
    // (modal save, tab switch, dim change, option change, page load, …)
    // also re-evaluates the button state. This matches the behaviour the
    // user confirmed was working previously.
    const hardNeeded = (tabId === 'sdp-tab') ? (lmiE > 0) : (n > 0);
    setHardConstraintButtonsEnabled(hardNeeded);
    const mode = currentMode;
    const option = (document.getElementById(tab + '-options') || {}).value || 'robust';
    const hasReg = (option === 'regularization' || option === 'regularization-relaxation');
    const hasRelax = (option === 'relaxation' || option === 'regularization-relaxation');

    // ── Scalar parameter sign checks, gated on the active option ──
    function checkNonNegatives(fieldId, label) {
        const el = document.getElementById(fieldId);
        const raw = el && el.value ? el.value.trim() : '';
        if (raw === '') return `${label} must be provided (comma-separated values ≥ 0).`;
        const parts = raw.split(',').map(s => s.trim()).filter(s => s !== '');
        if (parts.length === 0) return `${label} must be provided (comma-separated values ≥ 0).`;
        for (const part of parts) {
            const v = Number(part);
            if (!Number.isFinite(v)) return `${label} value "${part}" is not a number.`;
            if (v < 0) return `${label} value ${v} is negative — must be ≥ 0.`;
        }
        return '';
    }
    if (hasReg) {
        const err = checkNonNegatives(tab + '-tau', 'τ (regularization weight)');
        if (err) { setMsg(err); return {valid: false, message: err}; }
        // p is a single scalar (norm order). Reject lists, blanks-only, and
        // non-numeric / non-special values up front so the backend never
        // hits a cryptic float() ValueError.
        const pEl = document.getElementById(tab + '-p');
        const pRaw = (pEl ? pEl.value : '').trim().toLowerCase().replace(/["']/g, '');
        if (pRaw !== '') {
            const isList = pRaw.includes(',');
            const isSpecial = pRaw === 'inf' || pRaw === 'infinity' || pRaw === 'np.inf' || pRaw === 'fro';
            const asNum = Number(pRaw);
            const isNum = Number.isFinite(asNum);
            if (isList) {
                const msg = `p (norm order) must be a single value, not a list — you entered "${pEl.value}". Sweep over τ or ρ instead.`;
                setMsg(msg); return {valid: false, message: msg};
            }
            if (!isSpecial && !isNum) {
                const msg = `p (norm order) must be a number, "inf", or "fro" — you entered "${pEl.value}".`;
                setMsg(msg); return {valid: false, message: msg};
            }
            if (isNum && asNum < 1) {
                const msg = `p (norm order) value ${asNum} is less than 1 — p must be ≥ 1.`;
                setMsg(msg); return {valid: false, message: msg};
            }
        }
    }
    if (hasRelax) {
        const err = checkNonNegatives(tab + '-rho', 'ρ (relaxation penalty)');
        if (err) { setMsg(err); return {valid: false, message: err}; }
    }

    function parseJSON(id) {
        const el = document.getElementById(id);
        if (!el || !el.value) return null;
        try { return JSON.parse(el.value); } catch (e) { return {__parseError: true}; }
    }
    function shape(mat) {
        if (!Array.isArray(mat)) return [0, 0];
        const r = mat.length;
        const c = r > 0 && Array.isArray(mat[0]) ? mat[0].length : 0;
        return [r, c];
    }
    function checkShape(id, label, expectedRows, expectedCols) {
        const parsed = parseJSON(id);
        if (parsed === null) return `${label} is not set.`;
        if (parsed.__parseError) return `${label} has invalid JSON.`;
        const [r, c] = shape(parsed);
        if (r !== expectedRows || c !== expectedCols) {
            return `${label} is ${r}×${c} but expected ${expectedRows}×${expectedCols}.`;
        }
        return '';
    }
    // Optional-matrix variant: "empty" is OK. Used for G/h that are only
    // required when n > 0.
    function checkOptional(id, label, expectedRows, expectedCols) {
        const el = document.getElementById(id);
        if (!el || !el.value) return '';
        return checkShape(id, label, expectedRows, expectedCols);
    }

    if (d <= 0) { setMsg('Decision variable count d must be ≥ 1.'); return {valid: false, message: errEl ? errEl.textContent : ''}; }

    // ── LP / QP ──
    if (tabId === 'lp-tab' || tabId === 'qp-tab') {
        // c (d × 1) always required
        let err = checkShape(`${tab}-c`, 'c', d, 1);
        if (err) { setMsg(err); return {valid: false, message: err}; }
        if (tabId === 'qp-tab') {
            err = checkShape(`${tab}-Q`, 'Q', d, d);
            if (err) { setMsg(err); return {valid: false, message: err}; }
        }
        if (mode === 'symbolic') {
            if (m <= 0) { setMsg('Soft constraint row count 𝔪 must be ≥ 1 in symbolic mode.'); return {valid: false, message: 'm=0'}; }
            err = checkShape(`${tab}-A_d`, 'A(δ)', m, d);
            if (err) { setMsg(err); return {valid: false, message: err}; }
            err = checkShape(`${tab}-b_d`, 'b(δ)', m, 1);
            if (err) { setMsg(err); return {valid: false, message: err}; }
        }
        if (n > 0) {
            err = checkShape(`${tab}-G`, 'G', n, d);
            if (err) { setMsg(err); return {valid: false, message: err}; }
            err = checkShape(`${tab}-h`, 'h', n, 1);
            if (err) { setMsg(err); return {valid: false, message: err}; }
        } else {
            // If n == 0, G and h must not be set either (or must be empty).
            // Accept any stored value silently — backend ignores when n=0.
        }
        if (hasReg) {
            err = checkOptional(`${tab}-theta-bar`, 'x̄', d, 1);
            if (err) { setMsg(err); return {valid: false, message: err}; }
        }
    }
    // ── SDP ──
    else if (tabId === 'sdp-tab') {
        let err = checkShape('sdp-c', 'c', d, 1);
        if (err) { setMsg(err); return {valid: false, message: err}; }
        err = checkOptional('sdp-Q', 'Q', d, d);
        if (err) { setMsg(err); return {valid: false, message: err}; }

        if (mode === 'symbolic') {
            if (lmi <= 0) { setMsg('LMI size 𝔪 must be ≥ 1 in symbolic mode.'); return {valid: false, message: 'lmi=0'}; }
            const parsed = parseJSON('sdp-F_d');
            if (parsed === null) { setMsg('F(δ) collection is not set.'); return {valid: false, message: 'F missing'}; }
            if (parsed.__parseError) { setMsg('F(δ) collection has invalid JSON.'); return {valid: false, message: 'F bad json'}; }
            if (typeof parsed !== 'object' || Array.isArray(parsed)) {
                const msg = 'F(δ) must be a dict of matrices keyed "0"…"d".';
                setMsg(msg); return {valid: false, message: msg};
            }
            const expectedKeys = d + 1;
            const keys = Object.keys(parsed);
            if (keys.length !== expectedKeys) {
                const msg = `F(δ) collection has ${keys.length} matrices but expected ${expectedKeys} (one per j = 0…${d}).`;
                setMsg(msg); return {valid: false, message: msg};
            }
            for (let j = 0; j <= d; j++) {
                if (!(String(j) in parsed) && !(j in parsed)) {
                    const msg = `F(δ) is missing key "${j}".`;
                    setMsg(msg); return {valid: false, message: msg};
                }
                const mat = parsed[String(j)] !== undefined ? parsed[String(j)] : parsed[j];
                const [r, c] = shape(mat);
                if (r !== lmi || c !== lmi) {
                    const msg = `F_${j}(δ) is ${r}×${c} but expected ${lmi}×${lmi}.`;
                    setMsg(msg); return {valid: false, message: msg};
                }
            }
        }
        if (lmiE > 0) {
            const parsed = parseJSON('sdp-E');
            if (parsed === null) {
                const msg = 'E collection is not set (LMI-E size 𝔫 > 0).';
                setMsg(msg); return {valid: false, message: msg};
            }
            if (parsed.__parseError) { setMsg('E collection has invalid JSON.'); return {valid: false, message: 'E bad json'}; }
            const expectedKeys = d + 1;
            const keys = Object.keys(parsed);
            if (keys.length !== expectedKeys) {
                const msg = `E collection has ${keys.length} matrices but expected ${expectedKeys} (one per j = 0…${d}).`;
                setMsg(msg); return {valid: false, message: msg};
            }
            for (let j = 0; j <= d; j++) {
                const k = (String(j) in parsed) ? String(j) : j;
                const mat = parsed[k];
                const [r, c] = shape(mat);
                if (r !== lmiE || c !== lmiE) {
                    const msg = `E_${j} is ${r}×${c} but expected ${lmiE}×${lmiE}.`;
                    setMsg(msg); return {valid: false, message: msg};
                }
            }
        }
        if (hasReg) {
            const err2 = checkOptional('sdp-theta-bar', 'x̄', d, 1);
            if (err2) { setMsg(err2); return {valid: false, message: err2}; }
        }
    }

    // Final (non-optional) check: scenarios file must be uploaded. Placed
    // last so every other input issue surfaces first, but still mandatory.
    const scenarioFileInput = document.getElementById('file');
    if (!scenarioFileInput || !scenarioFileInput.files || scenarioFileInput.files.length === 0) {
        const msg = 'No scenarios file uploaded. Upload a scenario data file (CSV, JSON, MAT, …) under "Scenarios" before solving.';
        setMsg(msg);
        return {valid: false, message: msg};
    }

    // All good — clear any previous error.
    setMsg('');
    return {valid: true, message: ''};
}

function solve() {
    // Gate on pre-solve validation first; red banner appears above Solve.
    const check = validateSolveInputs();
    if (!check.valid) return;

    const beta = Number((document.getElementById('confidence') || {}).value);
    if (!(beta > 0 && beta < 1)) {
        const errEl = document.getElementById('solve-error');
        if (errEl) {
            errEl.textContent = 'Enter the confidence parameter β, a number between 0 and 1 (e.g. 1e-06).';
            errEl.style.display = 'block';
        }
        return;
    }

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

    // If n = 0 on the current tab, strip hard-constraint matrices from the
    // payload so the server never evaluates them as active constraints, even
    // if stale JSON lingers in the hidden inputs.
    const rowsGLive = parseInt(document.getElementById('dim-rows-g').value) || 0;
    const lmiELive = parseInt(document.getElementById('dim-lmi-e-size').value) || 0;
    if (activeTab === 'sdp-tab') {
        if (lmiELive === 0) formData.set('E', '');
    } else {
        if (rowsGLive === 0) { formData.set('G', ''); formData.set('h', ''); }
    }

    // Append mode and dimension controls
    formData.append('mode', currentMode);
    formData.append('n_x', document.getElementById('dim-nx').value);
    formData.append('rows_A', document.getElementById('dim-rows-a').value);
    formData.append('rows_G', document.getElementById('dim-rows-g').value);
    formData.append('lmi_size', document.getElementById('dim-lmi-size').value);
    formData.append('lmi_e_size', document.getElementById('dim-lmi-e-size').value);

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

    // Append MOSEK license if provided. Instance of File is required — if a
    // cached pseudo-File decoded wrong this will throw clearly, rather than
    // silently skipping the license and giving an opaque backend error.
    if (mosekLicenseFile) {
        if (!(mosekLicenseFile instanceof Blob)) {
            throw new Error('Internal: MOSEK license file is not a Blob/File object.');
        }
        formData.append('mosek_license', mosekLicenseFile, mosekLicenseFile.name || 'mosek.lic');
    }

    // Clear any stale pre-solve banner now that we're actually submitting.
    const solveErrBox = document.getElementById('solve-error');
    if (solveErrBox) { solveErrBox.textContent = ''; solveErrBox.style.display = 'none'; }

    document.getElementById('result-box').innerHTML = '<p>Solve button pressed, processing results...</p>';

    // On the hosted app, suggest a local install once a solve runs for a while.
    const isLocalHost = ['localhost', '127.0.0.1', '[::1]', ''].includes(window.location.hostname);
    const slowSolveTimer = isLocalHost ? null : setTimeout(function() {
        const box = document.getElementById('result-box');
        if (box) box.insertAdjacentHTML('beforeend',
            '<div class="alert alert-info mt-2">This problem is taking a while: problems of this size are better solved locally. ' +
            'See the <a href="https://github.com/Kiguli/Scen-Opt/wiki/Local_Install" target="_blank" rel="noopener">local install guide</a>.</div>');
    }, 30000);

    fetch(activeForm.action, {
        method: activeForm.method,
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => {
                    let message = 'Server error (' + response.status + '): ' + text;
                    try {
                        const body = JSON.parse(text);
                        if (body && body.error) message = body.error;
                    } catch (e) { /* not JSON: keep the raw text */ }
                    throw new Error(message);
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
            const option = (data.form_data && data.form_data.option) || 'robust';
            const showRhoTab = (option === 'relaxation' || option === 'regularization-relaxation');
            const showTauTab = (option === 'regularization' || option === 'regularization-relaxation');
            const resultBox = document.getElementById('result-box');
            resultBox.innerHTML = `
    <div class="result-container">
    <ul class="nav nav-tabs" id="resultTab" role="tablist">
        <li class="nav-item">
            <a class="nav-link active" id="result-table-tab" data-toggle="tab" href="#result-table" role="tab" aria-controls="result-table" aria-selected="true">Results Table</a>
        </li>
        ${showRhoTab ? `<li class="nav-item">
            <a class="nav-link" id="result-graph-tab" data-toggle="tab" href="#result-graph" role="tab" aria-controls="result-graph" aria-selected="false">ρ Graph</a>
        </li>` : ''}
        ${showTauTab ? `<li class="nav-item">
            <a class="nav-link" id="result-graph2-tab" data-toggle="tab" href="#result-graph2" role="tab" aria-controls="result-graph2" aria-selected="false">𝜏 Graph</a>
        </li>` : ''}
        <li class="nav-item">
            <a class="nav-link" id="result-formdata-tab" data-toggle="tab" href="#result-formdata" role="tab" aria-controls="result-formdata" aria-selected="false">Program JSON</a>
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
        ${showRhoTab ? `<div class="tab-pane fade" id="result-graph" role="tabpanel" aria-labelledby="result-graph-tab">
            <h3><br> ρ Graph</h3>
            <div class="form-group">
            <label for="tau-select">Select 𝜏:</label>
            <select id="tau-select" class="form-control" style="width: 200px; display: inline-block;"></select>
            </div>
            <canvas id="resultGraph" width="400" height="300"></canvas>
        </div>` : ''}
        ${showTauTab ? `<div class="tab-pane fade" id="result-graph2" role="tabpanel" aria-labelledby="result-graph2-tab">
            <h3><br> 𝜏 Graph</h3>
            <div class="form-group">
            <label for="rho-select">Select ρ:</label>
            <select id="rho-select" class="form-control" style="width: 200px; display: inline-block;"></select>
            </div>
            <canvas id="resultGraph2" width="400" height="300"></canvas>
        </div>` : ''}
        <div class="tab-pane fade" id="result-formdata" role="tabpanel" aria-labelledby="result-formdata-tab">
            <h3><br> Form Data
                <button class="btn btn-outline-success btn-sm ml-3" onclick="downloadProgramJSON()">Download Program (.JSON)</button>
            </h3>
            <p class="text-muted small mb-2">This is the exact JSON you can paste or upload via the <b>Upload Program</b> button at the top of the page to restore this problem in a future session.</p>
            <pre style="white-space: pre-wrap; word-break: break-all;">${JSON.stringify(formDataToProgramJSON(data.form_data), null, 2)}</pre>
        </div>
    </div>
</div>
`;

            // Initialise popovers inserted inside the freshly-rendered
            // results panel (e.g. the ? next to Optimal Cost). The global
            // .dim-help-btn popover registration runs only at page load, so
            // these dynamic buttons need their own hookup.
            $('#result-box .result-help-btn').popover({ trigger: 'hover', html: true });

            // Prepare risk lower and upper arrays
            // A degenerate run's lower bound is not certified, so it is left off the graphs (null = gap).
            const degenerateRuns = Array.isArray(data.degeneracy) ? data.degeneracy : [data.degeneracy];
            const riskLower = Array.isArray(data.risk)
                ? data.risk.map((r, i) => degenerateRuns[i] ? null : (Array.isArray(r) ? r[0] : r))
                : [degenerateRuns[0] ? null : data.risk];
            const riskUpper = Array.isArray(data.risk) ? data.risk.map(r => Array.isArray(r) ? r[1] : r) : [data.risk];

            // Assume data.rho_ and data.tau_ are arrays of equal length, and each (rho, tau) pair is unique
            const uniqueTau = Array.from(new Set(Array.isArray(data.tau_) ? data.tau_ : [data.tau_]));
            const tauSelect = document.getElementById('tau-select');
            if (tauSelect) {
                tauSelect.innerHTML = uniqueTau.map(tau => `<option value="${tau}">${tau}</option>`).join('');
                // Hide the selector + its label when there's only one value to
                // pick from — the dropdown has nothing to disambiguate.
                const tauWrap = tauSelect.closest('.form-group');
                if (tauWrap) tauWrap.style.display = uniqueTau.length > 1 ? '' : 'none';
            }

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
                const canvas = document.getElementById('resultGraph');
                if (!canvas) return;
                const filtered = getFilteredByTau(selectedTau);
                const ctx = canvas.getContext('2d');
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
                                tension: 0
                            },
                            {
                                label: 'Risk Upper Bound (εᵤ)',
                                data: filtered.riskUpper,
                                borderColor: 'rgba(54, 162, 235, 1)',
                                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                                yAxisID: 'y-risk',
                                fill: '-1',
                                tension: 0
                            },
                            {
                                label: 'Cost',
                                data: filtered.cost,
                                borderColor: 'rgba(36, 32, 34, 0.8)',
                                yAxisID: 'y-cost',
                                fill: false,
                                tension: 0
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
            if (showRhoTab) {
                drawRhoChart(uniqueTau[0]);
                if (tauSelect) tauSelect.addEventListener('change', e => drawRhoChart(e.target.value));
            }

            // Assume data.rho_ and data.tau_ are arrays of equal length, and each (rho, tau) pair is unique
            const uniqueRho = Array.from(new Set(Array.isArray(data.rho_) ? data.rho_ : [data.rho_]));
            const rhoSelect = document.getElementById('rho-select');
            if (rhoSelect) {
                rhoSelect.innerHTML = uniqueRho.map(rho => `<option value="${rho}">${rho}</option>`).join('');
                const rhoWrap = rhoSelect.closest('.form-group');
                if (rhoWrap) rhoWrap.style.display = uniqueRho.length > 1 ? '' : 'none';
            }

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
                const canvas2 = document.getElementById('resultGraph2');
                if (!canvas2) return;
                const filtered = getFilteredByRho(selectedRho);
                const ctx2 = canvas2.getContext('2d');
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
                                tension: 0
                            },
                            {
                                label: 'Risk Upper Bound (εᵤ)',
                                data: filtered.riskUpper,
                                borderColor: 'rgba(54, 162, 235, 1)',
                                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                                yAxisID: 'y-risk',
                                fill: '-1',
                                tension: 0
                            },
                            {
                                label: 'Cost',
                                data: filtered.cost,
                                borderColor: 'rgba(36, 32, 34, 0.8)',
                                yAxisID: 'y-cost',
                                fill: false,
                                tension: 0
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
            if (showTauTab) {
                drawTauChart(uniqueRho[0]);
                if (rhoSelect) rhoSelect.addEventListener('change', e => drawTauChart(e.target.value));
            }
            safeTypesetPromise();
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result-box').innerHTML =
                '<div class="alert alert-danger"><strong>Error:</strong> ' + error.message + '</div>';
        })
        .finally(() => clearTimeout(slowSolveTimer));
}

function generateResultTable(data) {
    // Get arrays of rho and tau
    const rhos = Array.isArray(data.rho_) ? data.rho_ : [data.rho_];
    const taus = Array.isArray(data.tau_) ? data.tau_ : [data.tau_];
    const count = Math.max(rhos.length, taus.length);

    // Generate headers
    const headers = Array.from({length: count}, (_, i) => `<th>Value${i + 1}</th>`).join('');

    // ── Display formatter: tiered numeric rendering.
    //    • |n| < 5e-5   → scientific (1.234e-5)        – avoids "0.0000"
    //    • 5e-5 ≤ |n| < 10 → 4 decimal places (0.1234, 9.8765)
    //    • 10 ≤ |n| < 1e7  → 2 decimal places (1234.58) – fewer trailing zeros
    //    • |n| ≥ 1e7   → scientific (1.234e+7)
    //    Strings, booleans, null, and non-numeric content pass through unchanged.
    //    Only affects rendering — `lastResultData` keeps full precision so
    //    the Download .JSON / Download .MAT buttons export unrounded numbers.
    function fmt(v) {
        if (v === null || v === undefined || v === '') return v;
        if (typeof v === 'boolean') return v;
        const isNumeric = typeof v === 'number'
            || (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v)));
        if (!isNumeric) return v;
        const n = Number(v);
        if (!Number.isFinite(n)) return v;
        if (n === 0) return '0.0000';
        const a = Math.abs(n);
        if (a < 5e-5 || a >= 1e7) return n.toExponential(3);
        if (a >= 10) return n.toFixed(2);
        return n.toFixed(4);
    }
    function fmtArr(arr) {
        return Array.isArray(arr) ? arr.map(fmt) : fmt(arr);
    }

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
        {label: `Optimal Cost <button type="button" class="btn btn-link p-0 ml-1 dim-help-btn result-help-btn"
                data-toggle="popover" data-trigger="hover" data-placement="right" data-html="true"
                data-content="<b>What is reported:</b><br>• LP: \\(c^\\top x^*\\)<br>• QP / SDP: \\(c^\\top x^* + \\tfrac{1}{2}\\,x^{*\\top} Q\\, x^*\\)<br><br>The regularization term \\(\\tau \\lVert x - \\bar{x}\\rVert_p\\) and the relaxation term \\(\\rho \\sum_i \\zeta_i\\) are <b>excluded</b>. They are treated as solver-guidance penalties, not part of the underlying problem's objective, so they do not leak into the reported cost."><span class="dim-help-icon">?</span></button>`, values: data.optimal_cost},
        {label: `Optimal x`, values: data.optimal_x},
        ...(hasRelaxation ? [{label: 'Optimal &zeta;', values: data.optimal_s}] : []),
        {label: 'Relaxation Parameters &rho;', values: data.rho_},
        {label: 'Regularization Parameters &tau;', values: data.tau_},
        {label: (function() {
            const opt = (data.form_data && data.form_data.option) || 'robust';
            if (opt === 'relaxation') return 'Confidence \\(1 - \\beta \\cdot n_{\\rho}\\)';
            if (opt === 'regularization') return 'Confidence \\(1 - \\beta \\cdot n_{\\tau}\\)';
            if (opt === 'regularization-relaxation') return 'Confidence \\(1 - \\beta \\cdot n_{\\tau} \\cdot n_{\\rho}\\)';
            return 'Confidence \\(1 - \\beta\\)';
        })(), values: data.conf},
        {label: `Risk Bounds &epsilon; <button type="button" class="btn btn-link p-0 ml-1 dim-help-btn result-help-btn"
                data-toggle="popover" data-trigger="hover" data-placement="right" data-html="true"
                data-content="Lower and upper bounds on the risk, each holding with the confidence above. The upper bound always holds. The lower bound needs non-degeneracy: when degeneracy is detected it is not certified and is shown as N/A."><span class="dim-help-icon">?</span></button>`, values: data.risk},
        {label: 'Degeneracy Detected?', values: data.degeneracy},
        {label: 'Complexity (support list size)', values: data.active_con},
        {label: 'Number of data samples', values: data.num_deltas},
        {label: `Total Constraints <button type="button" class="btn btn-link p-0 ml-1 dim-help-btn result-help-btn"
                data-toggle="popover" data-trigger="hover" data-placement="right" data-html="true"
                data-content="The \\(N\\) scenario constraints plus the hard constraints (the rows of \\(G\\), or the hard LMI)."><span class="dim-help-icon">?</span></button>`, values: data.tot_con},
        {label: `Optimization Time (s) <button type="button" class="btn btn-link p-0 ml-1 dim-help-btn result-help-btn"
                data-toggle="popover" data-trigger="hover" data-placement="right" data-html="true"
                data-content="Time to solve the program and compute its support list (which re-solves the program)."><span class="dim-help-icon">?</span></button>`, values: data.solve_time},
        {label: 'Risk Computation Time (s)', values: data.risk_time},
        {label: '<i>Error</i>', values: data.errorcode}
    ];

    // Size the table so every Value column is at least 120px wide — enough
    // for the widest single-line numeric format we produce (e.g. "1.234e-100"
    // including the cell padding). When (180 + count*120) exceeds the
    // container width, the outer .result-table-container scrolls horizontally;
    // otherwise the table fills the container and table-layout:fixed gives
    // each Value column an equal share of the remaining space.
    const PARAM_COL_PX = 180;
    const MIN_VALUE_COL_PX = 120;
    const tableMinWidth = PARAM_COL_PX + count * MIN_VALUE_COL_PX;
    let table = `<table class="result-table" style="min-width:${tableMinWidth}px;">
        <colgroup>
            <col style="width:${PARAM_COL_PX}px;">
            ${Array.from({length: count}, () => `<col style="min-width:${MIN_VALUE_COL_PX}px;">`).join('')}
        </colgroup>
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
                    // Optimal x can come back as [[x_0], [x_1], …] for column
                    // vectors; flatten one level so each entry is a number.
                    const flat = val.map(v => Array.isArray(v) ? v[0] : v);
                    val = `<div style="max-height:150px; overflow-y:auto;">${flat.map(fmt).join('<br>')}</div>`;
                }
            }

            if (row.label.includes('Risk Bounds') && Array.isArray(val) && val.length === 2) {
                const degenerate = getValue(data.degeneracy, i) === true;
                val = `<strong>[${degenerate ? 'N/A' : fmt(val[0])}, ${fmt(val[1])}]</strong>`;
            } else if (row.label.includes('Confidence')) {
                // Clamp at 0 — over-aggressive sweeps can push 1 − β·q below
                // zero, which is nonsensical to display. Reuse fmt() so a
                // tiny but positive confidence still prints in scientific
                // form rather than rounding to 0.0000.
                // Show enough digits that 1 − β is not rounded to 1.0000.
                const v = Math.max(0, Number(val));
                val = Number.isFinite(v)
                    ? (v > 0.9999 && v < 1 ? String(Number(v.toPrecision(12))) : fmt(v))
                    : val;
            } else if (row.label.includes('Optimal Cost')) {
                val = `<strong>${fmt(val)}</strong>`;
            } else if (row.label.includes('Error')) {
                val = `<i>${Array.isArray(val) ? val.join('<br>') : val}</i>`;
            } else if (row.label.includes('Degeneracy Detected?')) {
                val = val ? 'Yes' : 'No';
            } else if (
                row.label.includes('Relaxation Parameters') ||
                row.label.includes('Regularization Parameters') ||
                row.label.includes('Optimization Time') ||
                row.label.includes('Risk Computation Time')
            ) {
                val = fmt(val);
            }
            table += `<td>${val !== undefined ? val : ''}</td>`;
        }
        table += `</tr>`;
    });
    table += `</table>`;
    return table;
}

/**
 * Convert the raw submitted form data back into the "Upload Program" JSON
 * schema so it can be saved now and re-uploaded in a future session to
 * restore the exact same problem setup.
 *
 * Mapping notes:
 *   - active_tab ("lp-tab"/"qp-tab"/"sdp-tab") → `type` ("LP"/"QP"/"SDP").
 *   - Matrix hidden inputs store JSON strings; parse them to arrays.
 *   - "theta-bar" (hyphenated form name) → `x_ref` (the key loadProblemJSON reads).
 *   - Drop fields that loadProblemJSON doesn't consume to keep the JSON tidy.
 */
function formDataToProgramJSON(formData) {
    if (!formData) return {};
    const tabToType = {'lp-tab': 'LP', 'qp-tab': 'QP', 'sdp-tab': 'SDP'};
    const type = tabToType[formData.active_tab] || 'LP';
    const out = { type: type };
    if (formData.mode) out.mode = formData.mode;

    const parseIfSet = function(key) {
        const v = formData[key];
        if (v === undefined || v === null || v === '') return undefined;
        try { return JSON.parse(v); } catch (e) { return v; }
    };
    const assign = function(targetKey, srcKey) {
        const v = parseIfSet(srcKey);
        if (v !== undefined) out[targetKey] = v;
    };

    // Scenario matrices (symbolic mode)
    assign('A_d', 'A_d');
    assign('b_d', 'b_d');
    assign('F_d', 'F_d');

    // Hard constraints
    assign('G', 'G');
    assign('h', 'h');
    assign('E', 'E');

    // Cost / quadratic
    assign('c', 'c');
    assign('Q', 'Q');

    // Regularization reference point (hyphen-named form field → x_ref key)
    const xRef = parseIfSet('theta-bar') !== undefined ? parseIfSet('theta-bar') : parseIfSet('theta_bar');
    if (xRef !== undefined) out.x_ref = xRef;

    // Scalar parameters — keep as strings so loadProblemJSON parses them
    // the same way regardless of number of sweep values.
    ['rho', 'tau', 'confidence', 'p'].forEach(function(k) {
        if (formData[k] !== undefined && formData[k] !== '') out[k] = formData[k];
    });

    // Preserve the explicit formulation option (robust / regularization /
    // relaxation / regularization-relaxation) so a round-trip upload
    // doesn't have to re-infer it from rho/tau sweep strings.
    if (formData.option) out.option = formData.option;
    // Note: solver is NOT saved — the uploaded program should inherit the
    // user's current solver choice, not pin to a past one.

    // Dimension controls — useful when uploading in numeric mode, or when
    // matrix shapes alone don't pin down d / m / lmi_size.
    if (formData.n_x)      out.n_x      = parseInt(formData.n_x, 10);
    if (formData.rows_A)     out.rows_A     = parseInt(formData.rows_A, 10);
    if (formData.rows_G)     out.rows_G     = parseInt(formData.rows_G, 10);
    if (formData.lmi_size)   out.lmi_size   = parseInt(formData.lmi_size, 10);
    if (formData.lmi_e_size) out.lmi_e_size = parseInt(formData.lmi_e_size, 10);

    return out;
}

function downloadProgramJSON() {
    if (!lastResultData) return;
    const program = formDataToProgramJSON(lastResultData.form_data);
    const blob = new Blob([JSON.stringify(program, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'program.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
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
            latexSoft = `${Ax} \\leq 0, \\quad i = 1, \\ldots, N`;
            showTau();
        } else if (option === 'relaxation') {
            latexObj = `\\displaystyle\\min_{x,\\,\\zeta_i \\geq 0} \\quad c^\\top x${qTerm} + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i`;
            latexSoft = `${Ax} \\leq \\zeta_i, \\quad i = 1, \\ldots, N`;
            showRho();
        } else if (option === 'regularization-relaxation') {
            latexObj = `\\displaystyle\\min_{x,\\,\\zeta_i \\geq 0} \\quad c^\\top x${qTerm} + \\tau\\Vert x - \\bar{x}\\Vert_{p} + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i`;
            latexSoft = `${Ax} \\leq \\zeta_i, \\quad i = 1, \\ldots, N`;
            showTau(); showRho();
        }
    } else if (tab === 'sdp') {
        latexHard = 'E_0 + \\displaystyle\\sum_{j=1}^{d}x_jE_j \\preceq 0';
        if (option === 'robust') {
            latexObj = '\\displaystyle\\min_{x} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx';
            latexSoft = `${Fsum} \\preceq 0, \\quad i = 1, \\ldots, N`;
        } else if (option === 'regularization') {
            latexObj = '\\displaystyle\\min_{x} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx + \\tau\\Vert x - \\bar{x}\\Vert_{p}';
            latexSoft = `${Fsum} \\preceq 0, \\quad i = 1, \\ldots, N`;
            showTau();
        } else if (option === 'relaxation') {
            latexObj = '\\displaystyle\\min_{x,\\,\\zeta_i \\geq 0} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexSoft = `${Fsum} \\preceq \\zeta_i I, \\quad i = 1, \\ldots, N`;
            showRho();
        } else if (option === 'regularization-relaxation') {
            latexObj = '\\displaystyle\\min_{x,\\,\\zeta_i \\geq 0} \\quad c^\\top x + \\frac{1}{2}x^\\top Qx + \\tau\\Vert x - \\bar{x}\\Vert_{p} + \\rho \\displaystyle\\sum_{i=1}^{N} \\zeta_i';
            latexSoft = `${Fsum} \\preceq \\zeta_i I, \\quad i = 1, \\ldots, N`;
            showTau(); showRho();
        }
    }

    document.getElementById(`${tab}-latex`).innerHTML =
        `<div class="text-center"><p>\\(${latexObj}\\)</p><p>subject to:</p><p>\\(${latexSoft}\\)</p><p>\\(${latexHard}\\)</p></div>`;
    safeTypeset();
    updateConfidenceLabel(option);
    // Option changed → τ/ρ may now or no longer be required. Re-validate so
    // a previously-shown τ/ρ banner doesn't linger after switching to robust.
    if (typeof validateSolveInputs === 'function') validateSolveInputs();
    // Different option ⇒ potentially different cone requirements (e.g.
    // regularization adds SOCP/POW) ⇒ the solver allow-list may shrink/grow.
    if (typeof filterSolvers === 'function') filterSolvers();
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
        span.innerHTML = '\\(1 - \\beta \\cdot n_\\rho\\)';
    } else if (option === 'regularization') {
        span.innerHTML = '\\(1 - \\beta \\cdot n_\\tau\\)';
    } else {
        span.innerHTML = '\\(1 - \\beta \\cdot n_\\tau \\cdot n_\\rho\\)';
    }
    safeTypesetPromise();
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

    // Use plain Unicode glyphs (no LaTeX → no MathJax typeset) so this can
    // run on every τ/ρ keystroke without UI lag. The display reads
    // "n_τ = 3, n_ρ = 2" instead of the prettier MathJax-rendered subscript,
    // but the cost drops from ~20 ms+ to microseconds.
    let parts = [];
    if (hasTau) parts.push(`n_τ = ${nTau}`);
    if (hasRho) parts.push(`n_ρ = ${nRho}`);

    if (parts.length > 0) {
        counter.textContent = parts.join(', ');
        counter.style.display = 'block';
    } else {
        counter.style.display = 'none';
    }
}

function onModeChange() {
    document.querySelectorAll('.symbolic-only').forEach(el => {
        el.style.display = currentMode === 'symbolic' ? '' : 'none';
    });
    updateScenarioLabel();
    safeTypesetPromise();
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
    var content = '<b>Symbolic:</b> Enter constraint matrices as expressions of the components of a scenario (<code>delta[k]</code>). The matrices are then evaluated at each scenario as specified later.<br><br>' + numericDesc;
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
    safeTypesetPromise();
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

/**
 * Enable/disable the Edit G / Edit h buttons (and the SDP hard-LMI button)
 * based on the same n value the solve validator uses. Being explicit with
 * both the .disabled property and the attribute avoids any lingering state
 * in browsers that memoise the attribute separately.
 */
function setHardConstraintButtonsEnabled(enabled) {
    document.querySelectorAll('.hard-constraint-btn').forEach(function(btn) {
        btn.disabled = !enabled;
        if (enabled) {
            btn.removeAttribute('disabled');
            btn.classList.remove('btn-outline-secondary', 'disabled');
            btn.classList.add('btn-secondary');
        } else {
            btn.setAttribute('disabled', 'disabled');
            btn.classList.remove('btn-secondary');
            btn.classList.add('btn-outline-secondary');
        }
    });
}

/**
 * Back-compat wrapper: reads the authoritative n value (rows_G for LP/QP,
 * lmi_e for SDP) and toggles the Edit G / Edit h buttons accordingly. Kept
 * so existing callers keep working, but validateSolveInputs() is now the
 * single source of truth and also calls setHardConstraintButtonsEnabled().
 */
function updateHardConstraintState() {
    const activeTab = document.querySelector('.nav-link.active');
    const isSDP = activeTab && activeTab.id === 'sdp-tab';
    const nVal = isSDP
        ? parseInt(document.getElementById('dim-lmi-e-size').value) || 0
        : parseInt(document.getElementById('dim-rows-g').value) || 0;
    setHardConstraintButtonsEnabled(nVal > 0);
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
            ['A_d', 'b_d', 'G', 'h', 'c', 'Q', 'F_d', 'E'].forEach(function(field) {
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

        // Treat empty fields (null, "", [] or {}) as omitted, so a program can
        // list fields it does not use (e.g. "G": [] with no hard constraints).
        Object.keys(data).forEach(function(key) {
            const v = data[key];
            if (v === null || v === '' ||
                (Array.isArray(v) && v.length === 0) ||
                (typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === 0)) {
                delete data[key];
            }
        });

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

        // Determine the formulation option: prefer explicit `option` in the
        // program JSON; otherwise infer from rho/tau sweep strings.
        let option = 'robust';
        if (typeof data.option === 'string' &&
            ['robust', 'regularization', 'relaxation', 'regularization-relaxation'].includes(data.option)) {
            option = data.option;
        } else {
            const hasRho = data.rho !== undefined && parseFloat(data.rho) > 0;
            const hasTau = data.tau !== undefined && parseFloat(data.tau) > 0;
            if (hasRho && hasTau) option = 'regularization-relaxation';
            else if (hasRho) option = 'relaxation';
            else if (hasTau) option = 'regularization';
        }

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
            // Prefer matrix shapes when the matrices are present (symbolic);
            // fall back to the explicit scalar keys (n_x / rows_A / rows_G)
            // that the Program JSON carries for numeric uploads.
            const d = data.c ? nRows(data.c) : (data.n_x || 0);
            const m = data.A_d ? nRows(data.A_d) : (data.rows_A || 0);
            const n = data.G ? nRows(data.G) : (data.rows_G || 0);

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
            if (data.G) document.getElementById(prefix + '-G').value = JSON.stringify(data.G);
            if (data.h) document.getElementById(prefix + '-h').value = JSON.stringify(data.h);
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
            // Prefer E's matrix shape when present; fall back to the explicit
            // lmi_e_size scalar so numeric SDP uploads without E still set n.
            const lmiESize = (data.E && data.E["0"]) ? nRows(data.E["0"]) : (data.lmi_e_size || 0);

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
                document.getElementById('sdp-E').value = JSON.stringify(data.E);
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

/**
 * Per-solver cone capability map. Keys are CVXPY-known solver names; values
 * are the cone classes each can handle. The ordering LP → QP → SOCP → SDP
 * is deliberate: SDP-capable solvers implicitly handle everything weaker,
 * but we list each required cone explicitly so the check is a simple
 * subset test regardless of the problem. EXP / POW appear in a few
 * advanced cases (non-quadratic p-norms); most solvers don't support them.
 */
const SOLVER_CONE_SUPPORT = {
    'CLARABEL':   ['LP', 'QP', 'SOCP', 'SDP', 'EXP', 'POW'],
    'SCS':        ['LP', 'QP', 'SOCP', 'SDP', 'EXP', 'POW'],
    'MOSEK':      ['LP', 'QP', 'SOCP', 'SDP', 'EXP', 'POW'],
    'COPT':       ['LP', 'QP', 'SOCP', 'SDP', 'EXP'],
    'CVXOPT':     ['LP', 'QP', 'SOCP', 'SDP'],
    'SDPA':       ['LP', 'SDP'],
    'ECOS':       ['LP', 'SOCP', 'EXP'],
    'ECOS_BB':    ['LP', 'SOCP', 'EXP'],
    'GUROBI':     ['LP', 'QP', 'SOCP'],
    'CPLEX':      ['LP', 'QP', 'SOCP'],
    'XPRESS':     ['LP', 'QP', 'SOCP'],
    'SCIP':       ['LP', 'QP', 'SOCP'],
    'NAG':        ['LP', 'QP', 'SOCP'],
    'OSQP':       ['LP', 'QP'],
    'PROXQP':     ['LP', 'QP'],
    'PIQP':       ['LP', 'QP'],
    'DAQP':       ['LP', 'QP'],
    'HIGHS':      ['LP', 'QP'],
    'CBC':        ['LP'],
    'SCIPY':      ['LP'],
    'GLPK':       ['LP'],
    'GLPK_MI':    ['LP'],
    'PDLP':       ['LP']
};

/**
 * Compute the cones required by the currently-active problem configuration.
 * Returns a Set of cone identifiers: 'LP', 'QP' (quadratic cost),
 * 'SOCP' (second-order-cone: 2-norm / Frobenius / quadratic constraint),
 * 'SDP' (positive-semidefinite), 'EXP' (exponential), 'POW' (power cone).
 * Relaxation adds no extra cone (just linear slacks); regularization's
 * contribution depends on the p-norm.
 */
function requiredCones() {
    const cones = new Set(['LP']);
    const activeTab = document.querySelector('.nav-link.active');
    if (!activeTab) return cones;
    const tabId = activeTab.id;
    const tab = tabId.replace('-tab', '');
    if (tabId === 'qp-tab') cones.add('QP');   // 1/2 x^T Q x in the objective
    if (tabId === 'sdp-tab') {
        cones.add('SDP');
        // The 1/2 x^T Q x term is only added to an SDP when Q has a nonzero entry.
        let Q = [];
        try { Q = JSON.parse((document.getElementById('sdp-Q') || {}).value || '[]'); } catch (e) { Q = [[1]]; }
        const nonzero = v => Array.isArray(v) ? v.some(nonzero) : Number(v) !== 0;
        if (nonzero(Q)) cones.add('QP');
    }
    const option = (document.getElementById(tab + '-options') || {}).value || 'robust';
    if (option === 'regularization' || option === 'regularization-relaxation') {
        const pRaw = ((document.getElementById(tab + '-p') || {}).value || '')
            .trim().toLowerCase().replace(/["']/g, '');
        if (!pRaw) {
            cones.add('SOCP');  // default p = 2
        } else if (pRaw === 'inf' || pRaw === 'infinity' || pRaw === 'np.inf') {
            // L_inf is LP-representable via linear constraints — no extra cone.
        } else if (pRaw === 'fro') {
            cones.add('SOCP');  // Frobenius norm over a vector ≡ 2-norm
        } else {
            const pNum = Number(pRaw);
            if (!Number.isFinite(pNum)) {
                cones.add('SOCP');
            } else if (pNum === 1) {
                // L_1 is LP-representable.
            } else if (pNum === 2) {
                cones.add('SOCP');
            } else {
                // Rational p-norms other than 1, 2, ∞ need the power cone.
                cones.add('POW');
            }
        }
    }
    return cones;
}

function filterSolvers() {
    const activeTab = document.querySelector('.nav-link.active');
    if (!activeTab) return;

    const tabTypeMap = {'lp-tab': 'LP', 'qp-tab': 'QP', 'sdp-tab': 'SDP'};
    const problemType = tabTypeMap[activeTab.id];
    if (!problemType) return;

    const solverSelect = document.getElementById('solver');
    const currentValue = solverSelect.value;
    const needed = requiredCones();
    let firstVisible = null;

    Array.from(solverSelect.options).forEach(option => {
        const types = (option.getAttribute('data-types') || '').split(',');
        const typeOK = types.includes(problemType);

        // Cone-subset check: every required cone must appear in the solver's
        // supported list. Unknown solvers (not in the map) pass through — we
        // prefer false-positive visibility over hiding a valid choice.
        const caps = SOLVER_CONE_SUPPORT[option.value];
        // A quadratic cost can also be handled with second-order cones.
        const coneOK = caps === undefined
            ? true
            : Array.from(needed).every(c => caps.includes(c) || (c === 'QP' && caps.includes('SOCP')));

        if (typeOK && coneOK) {
            option.hidden = false;
            option.style.display = '';
            option.disabled = false;
            if (!firstVisible) firstVisible = option.value;
        } else {
            // `hidden` is the HTML5-spec attribute Chrome actually honours
            // for <option>; style:none alone isn't reliable on initial paint.
            option.hidden = true;
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
        safeTypesetPromise();
    });

    // Filter solvers and update dimension visibility on tab change
    $('a[data-toggle="tab"]').on('shown.bs.tab', function() {
        updateModeHelpContent();
        relocateSharedElements();
        filterSolvers();
        updateDimensionVisibility();
        updateLatexText(this.id.replace('-tab', ''));
        updateScenarioLabel();
        validateSolveInputs();
    });

    // Re-validate whenever the formulation option changes (toggles which
    // matrices are required, e.g. x̄ when regularization is active).
    ['lp', 'qp', 'sdp'].forEach(function(tab) {
        const sel = document.getElementById(tab + '-options');
        if (!sel) return;
        sel.addEventListener('change', function() {
            // Reset all cost-function parameters that meaningfully depend on
            // the formulation: τ, ρ, p, x̄. They'd otherwise propagate stale
            // values across robust ↔ regularization ↔ relaxation switches.
            ['tau', 'rho', 'p', 'theta-bar'].forEach(function(field) {
                const el = document.getElementById(tab + '-' + field);
                if (el) el.value = '';
            });
            // Make sure the sweep counter and any stale banner are refreshed.
            updateSweepCounter();
            const box = document.getElementById('solve-error');
            if (box) { box.textContent = ''; box.style.display = 'none'; }
        });
    });

    // Re-validate as soon as a scenarios file is picked (or cleared).
    const scenarioFileEl = document.getElementById('file');
    if (scenarioFileEl) {
        ['change', 'input'].forEach(function(evt) {
            scenarioFileEl.addEventListener(evt, validateSolveInputs);
        });
    }

    // Initial filter on page load
    filterSolvers();
    updateDimensionVisibility();
    updateLatexText('lp');
    updateScenarioLabel();
    validateSolveInputs();

    // τ/ρ inputs only update the (cheap) sweep counter live. They do NOT
    // run validateSolveInputs() — the validator parses every matrix's JSON
    // and is too expensive per-keystroke. The Solve press still runs the
    // full check, so an invalid value can't sneak through.
    ['lp', 'qp', 'sdp'].forEach(function(tab) {
        var rhoEl = document.getElementById(tab + '-rho');
        var tauEl = document.getElementById(tab + '-tau');
        var handler = function() {
            updateSweepCounter();
            // Hide stale solve-error banner so it doesn't sit there after
            // the user has just fixed a flagged value.
            var box = document.getElementById('solve-error');
            if (box && box.style.display !== 'none') {
                box.textContent = ''; box.style.display = 'none';
            }
        };
        ['input', 'change'].forEach(function(evt) {
            if (rhoEl) rhoEl.addEventListener(evt, handler);
            if (tauEl) tauEl.addEventListener(evt, handler);
        });
    });

    // p-norm input has no live listeners — filterSolvers() runs only on
    // tab switch, option change, and Solve press. Re-filtering on every
    // keystroke (which is what the listener used to do) added perceptible
    // typing lag because filterSolvers walks every option in the dropdown.
    // Trade-off: if the user types a p value that needs a solver they
    // don't have, the Solve will simply fail at solve-time rather than
    // grey out the entry up-front.

    // Reshape stored matrices, refresh Edit G / Edit h toggle, and re-run
    // the solve validator on every dimension change. Each piece is cheap on
    // its own (no MathJax) — dim inputs are typed infrequently, so the
    // per-keystroke cost is acceptable here.
    function dimChanged() {
        reshapeAllStoredMatrices();
        updateHardConstraintState();
        validateSolveInputs();
    }
    const dimInputIds = ['dim-nx', 'dim-rows-a', 'dim-rows-g', 'dim-lmi-size', 'dim-lmi-e-size'];
    const dimInputIdSet = new Set(dimInputIds);
    dimInputIds.forEach(function(id) {
        const el = document.getElementById(id);
        if (!el) return;
        ['input', 'change', 'keyup', 'wheel', 'mouseup'].forEach(function(evt) {
            el.addEventListener(evt, dimChanged);
        });
    });
    // Safety net delegated listener in case an input is ever re-created.
    ['input', 'change'].forEach(function(evt) {
        document.addEventListener(evt, function(e) {
            if (e.target && dimInputIdSet.has(e.target.id)) dimChanged();
        }, true);
    });


    // Initialize MOSEK cache indicator
    updateMosekCacheIndicator();

    // MOSEK license modal: upload & solve (cache for future use).
    // Defensive: hide the modal, then dispatch the solve in its own try-
    // block. Earlier the whole handler could die silently if anything in
    // cacheMosekLicense's FileReader chain threw (e.g. sessionStorage full
    // in private browsing), leaving the user staring at a dead modal.
    document.getElementById('mosek-license-confirm').addEventListener('click', function() {
        try {
            const fileInput = document.getElementById('mosek-license-file');
            if (!fileInput || !fileInput.files || !fileInput.files.length) {
                alert('Please select a MOSEK license file.');
                return;
            }
            const licenseFile = fileInput.files[0];
            $('#mosekLicenseModal').modal('hide');
            fileInput.value = '';
            // Fire the solve unconditionally; caching happens in parallel and
            // its result is optional. Any exception propagates to the outer
            // try-catch instead of being swallowed by a dropped Promise.
            cacheMosekLicense(licenseFile).catch(function(err) {
                console.warn('MOSEK license caching failed (continuing):', err);
            });
            executeSolve(licenseFile);
        } catch (err) {
            console.error('MOSEK upload-and-solve handler threw:', err);
            const box = document.getElementById('solve-error');
            if (box) {
                box.textContent = 'MOSEK upload-and-solve failed: ' + (err && err.message ? err.message : err);
                box.style.display = 'block';
            } else {
                alert('MOSEK upload-and-solve failed: ' + err);
            }
        }
    });

    // MOSEK license modal: skip (local license installed)
    document.getElementById('mosek-license-skip').addEventListener('click', function() {
        document.getElementById('mosek-license-file').value = '';
        $('#mosekLicenseModal').modal('hide');
        executeSolve(null);
    });
});