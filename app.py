from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import json as json_module
import io
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from src.Miscellaneous import get_solvers
from src.solve_form import solve_form

app = Flask(__name__)


@app.route('/')
def index():
    solvers = get_solvers()
    active_tab = request.args.get('active_tab', 'lp-tab')  # Default to 'lp-tab' if not provided
    return render_template('index.html', solvers=solvers, active_tab=active_tab)


def parse_uploaded_file(file):
    """Parse an uploaded file into a Python object (list/dict/array).
    Supports JSON, CSV, TXT, NPY, NPZ, MAT, Excel, and Parquet."""
    filename = file.filename.lower()
    if filename.endswith('.json'):
        return json_module.load(file)
    elif filename.endswith('.csv') or filename.endswith('.txt') or filename.endswith('.tsv'):
        content = file.read().decode('utf-8')
        sep = '\t' if filename.endswith('.tsv') else ','
        rows = content.strip().split('\n')
        return [row.split(sep) for row in rows]
    elif filename.endswith('.npy'):
        return np.load(file).tolist()
    elif filename.endswith('.npz'):
        data = np.load(file)
        keys = list(data.keys())
        return data[keys[0]].tolist()
    elif filename.endswith('.mat'):
        import scipy.io
        file_bytes = io.BytesIO(file.read())
        mat = scipy.io.loadmat(file_bytes, squeeze_me=False)

        def mat_unwrap(v):
            """Recursively unwrap numpy types to plain Python."""
            if isinstance(v, np.ndarray):
                # Struct array → dict
                if v.dtype.names is not None:
                    item = v.flat[0] if v.size == 1 else v
                    sub = {}
                    for name in v.dtype.names:
                        key = name[1:] if name.startswith('M') and name[1:].isdigit() else name
                        sub[key] = mat_unwrap(item[name])
                    return sub
                # Scalar (1×1) string or char array → plain string
                if v.dtype.kind in ('U', 'S'):
                    return str(v.flat[0]) if v.size == 1 else str(v.flat[0])
                # Scalar (1×1) numeric → float
                if v.dtype.kind in ('f', 'i', 'u') and v.size == 1:
                    return float(v.flat[0])
                # Object (cell) array → recurse into each element
                if v.dtype == object:
                    # 2-D cell array → always preserve as list of lists
                    if v.ndim == 2:
                        return [[mat_unwrap(v[r, c]) for c in range(v.shape[1])] for r in range(v.shape[0])]
                    # 1-D cell array → list
                    if v.ndim == 1:
                        return [mat_unwrap(x) for x in v]
                    # 0-D cell → unwrap
                    return mat_unwrap(v.item())
                # Numeric 2-D+ array → nested lists
                return v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                return float(v)
            elif isinstance(v, str):
                return v
            return v

        result = {}
        for k, v in mat.items():
            if k.startswith('__'):
                continue
            result[k] = mat_unwrap(v)
        return result
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        import pandas as pd
        file_bytes = io.BytesIO(file.read())
        df = pd.read_excel(file_bytes, header=None)
        return df.values.tolist()
    elif filename.endswith('.parquet'):
        import pandas as pd
        file_bytes = io.BytesIO(file.read())
        df = pd.read_parquet(file_bytes)
        return df.values.tolist()
    else:
        raise TypeError(f"Unsupported file format: {filename}")


@app.route('/parse-file', methods=['POST'])
def parse_file():
    """Parse an uploaded file and return its content as JSON.
    Used by frontend modals to handle binary file formats."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    try:
        result = parse_uploaded_file(file)
        return jsonify({"data": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/download-mat', methods=['POST'])
def download_mat():
    """Convert JSON result data to a MATLAB .mat file and return it."""
    import scipy.io
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    def _none_to_nan(v):
        # A failed run has no result (None), which savemat cannot store.
        if v is None:
            return np.nan
        if isinstance(v, list):
            return [_none_to_nan(item) for item in v]
        return v

    mat_dict = {}
    for key, value in data.items():
        value = _none_to_nan(value)
        try:
            mat_dict[key] = np.array(value)
        except (ValueError, TypeError):
            mat_dict[key] = np.array(value, dtype=object)

    buf = io.BytesIO()
    scipy.io.savemat(buf, mat_dict)
    buf.seek(0)
    return send_file(buf, mimetype='application/x-matlab',
                     as_attachment=True, download_name='results.mat')


@app.route('/solve', methods=['POST'])
def solve():
    # Check if a MOSEK license was uploaded — if so, run the solve in a
    # fresh subprocess so that MOSEKLM_LICENSE_FILE is read before MOSEK
    # is imported.  Each request gets a UUID-based /tmp directory so
    # concurrent users never conflict.
    has_license = (
        'mosek_license' in request.files
        and request.files['mosek_license'].filename != ''
    )

    if not has_license:
        try:
            return _solve_inner()
        except (ValueError, TypeError) as error:
            return jsonify({"error": str(error)}), 400

    # ---- MOSEK subprocess path ----
    # Use a platform-neutral temp directory (tempfile.mkdtemp → %TEMP% on
    # Windows, /tmp on Linux/macOS). Hard-coding /tmp broke Windows hosts
    # because the directory may not exist and isn't writable for the user.
    tmp_dir = tempfile.mkdtemp(prefix=f'mosek_{uuid.uuid4().hex}_')
    license_path = os.path.join(tmp_dir, 'mosek.lic')

    try:
        # Save uploaded license to the ephemeral /tmp directory
        request.files['mosek_license'].save(license_path)
        if os.path.getsize(license_path) == 0:
            return jsonify({"error": "Uploaded MOSEK license file is empty."}), 400

        # Parse scenario file (if any) in-process so we can serialise it
        scenarios_list = None
        if 'file' in request.files and request.files['file'].filename != '':
            parsed = parse_uploaded_file(request.files['file'])
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list) and len(v) > 0:
                        scenarios_list = np.array(v, dtype=float).tolist()
                        break
                if scenarios_list is None:
                    return jsonify({"error": "MAT file does not contain a recognizable scenario matrix."}), 400
            else:
                scenarios_list = np.array(parsed, dtype=float).tolist()

        # Build JSON payload for the subprocess
        payload = json_module.dumps({
            "form": request.form.to_dict(),
            "scenarios": scenarios_list,
        })

        # Run the solve in a fresh Python process with the license set
        proc = subprocess.run(
            [sys.executable, '-m', 'src.mosek_solve', license_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=7200,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            return jsonify({"error": f"MOSEK subprocess failed: {stderr}"}), 500

        result = json_module.loads(proc.stdout)
        return jsonify(result)

    except subprocess.TimeoutExpired:
        return jsonify({"error": "MOSEK solve timed out."}), 504
    except json_module.JSONDecodeError:
        return jsonify({"error": "MOSEK subprocess returned invalid output."}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _solve_inner():
    # Handle scenarios file if present
    scenarios = None
    if 'file' in request.files and request.files['file'].filename != '':
        parsed = parse_uploaded_file(request.files['file'])
        if isinstance(parsed, dict):
            # MAT files return a dict; use the first array-like value
            for k, v in parsed.items():
                if isinstance(v, list) and len(v) > 0:
                    scenarios = np.array(v, dtype=float)
                    break
            if scenarios is None:
                raise TypeError("MAT file does not contain a recognizable scenario matrix.")
        else:
            scenarios = np.array(parsed, dtype=float)

    return jsonify(solve_form(request.form.to_dict(), scenarios))


if __name__ == '__main__':
    app.run(debug=True)
