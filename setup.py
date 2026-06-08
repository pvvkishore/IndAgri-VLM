from setuptools import setup, find_packages

setup(
    name="indagri_vlm",
    version="1.0.0",
    author="Dr. P.V.V. Kishore",
    author_email="pvvkishore@kluniversity.in",
    description=(
        "IndAgri-VLM: A Telugu-Grounded Multimodal Vision-Language Model "
        "for South Indian Commercial Crop Disease Diagnosis"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pvvkishore/IndAgri-VLM",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
