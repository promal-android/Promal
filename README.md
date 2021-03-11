# ProMal

Implementation of the paper *ProMal: Precise Window Transition Graphs for Android via Synergy of Program Analysis and Machine Learning*.


## Introduction

In this work, we focus on creating a graphical user interface (GUI) model for an Android app, i.e., a window transition graph (WTG).  

We propose ProMal, a “tribrid analysis” approach to construct a WTG for an app that synergistically combines static/dynamic program analysis and machine learning techniques.

## Requirements

+ Android SDK Version: 23
+ JAVA Version: 1.8.0_181
+ Python Version >= 3.6.0
+ Python Library:
	+ numpy >= 1.18.4
	+ Keras >= 2.3.1
	+ scikit-learn >= 0.22.2

## Usage

ProMal contains three major components.

1. The first step is to perform static WTG analysis on Android apps with **GATOR**, a state-of-the-art program analysis toolkit for Android. The static WTG analysis will generate a *.wtg.dot* file for each apk, which will further be used as input in the following steps. The **GATOR** toolkit and detailed instruction is available on their [website](http://web.cse.ohio-state.edu/presto/software/gator/). Also, we provide a copy of **GATOR** used in our project and related scripts in [Static WTG Analysis](StaticWTGAnalysis).  

2. The second step is to perform dynamic WTG analysis on Android apps with **Paladin**, a state-of-the-art app exploration tool. Detailed instruction is shown in [Dynamic WTG Analysis](DynamicWTGAnalysis).  

3. The third step is to perform window transition prediction based on the result of static and dynamic WTG analysis. This step mainly focuses on predicting the unverified transitions in the static WTG (those who cannot be matched in the dynamic WTG). Detailed instruction is shown in [Window Transition Prediction](WindowTransitionPrediction).  

## Acknowledgement

[GATOR](http://web.cse.ohio-state.edu/presto/software/gator/)  
[Paladin](https://github.com/pkuoslab/Paladin)
