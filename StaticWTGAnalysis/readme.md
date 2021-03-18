# StaticWTGAnalysis

The first step of Promal is to obtain Window Transition Graph(WTG) using static analysis. We use the state-of-the-art tool **GATOR** to do so.

## Introduction

This folder contains a copy of **GATOR** and the scripts to run it.
+ `gatorPromal/run_gator.py`: Run gator and collect WTG.

## Requirements

+ Python>=3.6.9
+ JDK Version: 1.8.0_265

## Usage

First, navigate to `./gatorPromal`.

Run the script:

```bash
    python3 run_gator.py --apk_dir <APK_DIR> --adk_dir <ANDROID_SDK_DIR>
```
The results will be written to `dot_output`.
