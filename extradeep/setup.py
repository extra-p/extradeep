from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

info = {}
with open("extradeep/__init__.py") as fp:
    exec(fp.read(), info)

setup(
    name="extradeep",
    version=info['__version__'],
    packages=find_packages(include=('extradeep', 'extradeep.*')),
    author="Extra-Deep project",
    description=info['__description__'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    entry_points={
        "console_scripts": [
            "extradeep = extradeep.extradeep.extradeep:main",
            "extradeep-instrumenter = extradeep.extradeep.extradeep_instrumenter:main",
            "extradeep-plotter = extradeep.extradeep.extradeep_plotter:main",
        ],
        "gui_scripts": [
            "extradeep-gui = extradeep.extradeep.extradeep_gui:main",
        ]
    },
    python_requires='>=3.7',

    install_requires=["pyside2~=5.13", "numpy~=1.18", "matplotlib~=3.2", "tqdm~=4.47", "pycubexr~=1.1",
                      "marshmallow~=3.7", "marshmallow_enum", "packaging~=20.0", "pyobjc-framework-Cocoa~=6.2; sys_platform == 'darwin'", "tabulate", "rich"]

)
