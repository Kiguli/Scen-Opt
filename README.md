To install you should install Python (version 3.10 is the one I have been using and works fine), you also need to install Pip (something like `sudo apt-get install python3-pip` depending on your operating system). Then in a terminal by navigating to the main directory and running `pip install -r requirements.txt` will install all the necessary prerequisites for the tool. Then running the command `python3 app.py` will start the tool and show a command similar to the following:

```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 778-464-535
127.0.0.1 - - [18/Sep/2025 16:22:08] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [18/Sep/2025 16:22:08] "GET /static/css/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [18/Sep/2025 16:22:08] "GET /static/js/main.js HTTP/1.1" 304 -

```

Clicking the webpage `http://127.0.0.1:5000` will open the tool.

In the benchmarks folder the following have been tested and work: `half_width_1d`, `iris_minimal_3d`, and `growth_bound_12d`. 

Current extensions still planned are as follows:

- Single load of entire configuration file
- I am confident LP and QP work completely, so far the SDP benchmarks I have run have not found optimal solutions, this needs more significant testing.
