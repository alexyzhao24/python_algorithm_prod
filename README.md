### Project setup

1) Project Structure
Root
├── python_algo_production/        # Main package (actual source code lives here)
│   ├── __init__.py
│   ├── algo/
│   ├── framework/
│   ├── lib/
│   └── utils/
├── test/               # All pytest code
│   ├── __init__.py
│   ├── test_*.py
│   └── test_lralgo.py
├── test_main/          # Independent python code without pytest
│   ├── __init__.py
│   ├── opencv_dnn_test.py
│   └── v4l2py_test.py
│   └── v4l2py_filter_test.py
├── model/              # Optional for development only, not for release
    |__ model.pkl
├── .git                # GitHub Actions configuration
│── requirements.txt    # empty for now
├── setup.py            # key setup file
├── README.md           # This file
├── .gitignore          # Specifies intentionally untracked files to ignore
└── MAINFEST.in         # required for package build (wheel or sdist).

2) Editable installation without Manual path management
    * With editable package installation, -e or --editable flag, you tell pip to create a symbolic link from the installed package to the source code in your local directory. This means any changes you make to the source code are immediately reflected without needing to reinstall the package.

    * Create a basic setup.py
        from setuptools import setup, find_packages
        setup(
            name="python_algo_production",
            version="0.1",
            packages=find_packages(),
            package_data={"python_algo_production.lib": ["*.so"]},  # For installed packages (wheels, installs)
            include_package_data=True,                              # Needs MAINIFEST.in! for source distribution (sdist)
            install_requires=[],
        )

    * Execute ediable installation locally for easy module import
        pip install -e .
            Successfully installed python_algo_production-0.1

            This will create the necessary files under the project root:
            > ls -l python_algo_production.egg-info/
            -rw-rw-r-- 1 xxx xxx   1 Mar 10 12:36 dependency_links.txt
            -rw-r--r-- 1 xxx xxx  64 Mar 10 12:36 PKG-INFO
            -rw-rw-r-- 1 xxx xxx 355 Mar 10 12:36 SOURCES.txt
            -rw-rw-r-- 1 xxx xxx  23 Mar 10 12:36 top_level.txt


### Integration into VS Code

1) Ensure pytest is Enabled
    Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on macOS).
    Search for and select Python: Configure Tests.
	--> Choose pytest as your test framework.
	--> Select the folder containing your tests: test (not root, not python_algo_production)

2) Run Tests
    If you have a project with tests, the Test Explorer view discovers and lists the tests in your workspace. By default, the discovered tests are displayed in a tree view in the Test Explorer. The tree view matches the hierarchical structure of your tests, making it easy to navigate and run your tests.

    Click on the Testing icon (chemical vase) in the Activity Bar to open the Testing view.
        In this view, you should see your pytest tests listed if they've been discovered.
        * You can run individual tests, test files, or all tests using the play buttons next to each item.
        * To run or debug tests, you can also use the Run and Debug buttons at the top of the Testing view.

    To debug tests, you can select Debug buttons at the top of the Testing view.
        * Special about pytest debug defined in launch.json
            > Add the following line into the launch.json used for debugging
                    "purpose": ["debug-test"]: tells VS Code to use this config when you choose Debug Test
                    Otherwise, you will likely run into an error of calling ~/miniconda3/lib/python312.zip
            > "type": "python",   // use "python" for pytest debugging: "debuggy" for regular python file will not work here.

    ============================= test session starts ==============================
    platform linux -- Python 3.12.4, pytest-8.2.2, pluggy-1.5.0
    rootdir: /home/xxx/projects/python_algo_production
    configfile: pytest.ini
    plugins: dash-2.18.2, anyio-4.4.0, devtools-0.12.2
    collected 1 item

### Run Pytest

This project use pytest for testing algorithms.

0) pytest.ini
[pytest]
addopts = --import-mode=importlib

Why use --import-mode=importlib?
* Fix confusing import errors
If you see: ImportError: attempted relative import with no known parent package, this may help.

* Improves debugging in IDEs
Debuggers (like in VS Code or PyCharm) behave better because module names and locations are consistent.

* Match normal Python behavior
Makes pytest import behavior match python -m mymodule behavior.
Ensures __name__ and __package__ are correct in tests.


1) To run all tests in a directory: -v show the list of tests in a module
    pytest test
    pytest -v test

2) To run all tests in a module: -v show the list of tests in a module
    pytest test/test_lralgo.py
    pytest -v test/test_lralgo.py

3) To run a specfic test in a module:
    pytest -v test/test_lralgo.py::test_lr_hist_naive_basic
    pytest -v test/test_lralgo.py::test_lr_hist_basic

4) To run multiple selected tests in a module that share the same figure:
    pytest test/test_euro_filter.py::test_two_euro_filter_step test/test_euro_filter.py::test_two_euro_filter

5) To enable print that is captured by pytest for being clean
    pytest -s test/test_euro_filter.py

6) You can use pytest to invoke independent code without any pytesting :-)
    pytest test_main/v4l2py_test.py



### Pytest Debug Code

1) Use --pdb to trigger Debugger ONLY on Test Failures
    pytest --pdb test/test_lralgo.py::test_lr_hist_naive_basic

2) Use --trace to Debug from the Start
    pytest --trace test/test_lralgo.py::test_lr_hist_naive_basic


### How to use *.so files
1) About MAINFEST.in
    ✅ Include the .so file in your package build (wheel or sdist).

2) About setup.py :
    ✅ Install the .so file into your package directory (e.g., site-packages/python_algo_production/lib/*.so).

3) To import the library into your code:
    from python_algo_production.libs import *.so

### About Models

1) Use srcnn_export_models.py
    This code load in a pytorch model and convert it first into an onnx model, and finally Tensor RT model

2) Tensor RT model
    Make sure that Tensor RT model matched the available Tensor RT version.

    Failed example:
        * SRCNN_dynamic.engine was generated earlier based on Tensor RT 10.10
        * pip installed the latest Tensor RT 10.11.0
        * Run opencv_dnn_test.py to load in this "old model":
            TensorRT Python file: ~/miniconda3/lib/python3.12/site-packages/tensorrt/__init__.py
            TensorRT Python version: 10.11.0.33
            [06/16/2025-09:57:23] [TRT] [E] IRuntime::deserializeCudaEngine: Error Code 1: Serialization (Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed.Version tag does not match. Note: Current Version: 240, Serialized Engine Version: 239)
    Solution: Rengerate the model based on Tensor RT 10.11.0