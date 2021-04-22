# ProMal

Implementation of the paper *ProMal: Precise Window Transition Graphs for Android via Synergy of Program Analysis and Machine Learning*.


## Introduction

Promal is a novel tool we develop to build a more precise static GUI model of Android Apps (i.e., a Window Transition Graph, or WTG for short). A WTG models how windows transit to each other when a certain action is triggered with nodes being windows and edges being widgets and action on which that triggers the transition. The correctness of the WTG is critical for a variaty of automatic app analysis tasks. Existing work based on static analysis or dynamic analysis can not address the problem of over-approximation or low coverage. 

ProMal is a “tribrid analysis” approach to construct a WTG for an app that synergistically combines static/dynamic program analysis and machine learning techniques. Promal utilizes static analysis to build a base WTG and verifies the edges using dynamic analysis. The edges cannot be verified are predicted using a machine learning module.

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

## Appendix
List of the APPs used for evaluation
| **App ID** | Package Name            | Density | LOC | Category | Group |
|:---------------|:----------------------------------|------------------:|--------------:|-------------------|------------|
| App 1         | com.jacobsmedia.sparts           | 1.50             | 6,205        | Sports            |1|
| App 2         | com.galaxyodyssey.pushgenius     | 2.50             | 1,815        | Game              |1|
| App 3         | com.ajas.CoinFlipFullFree        | 2.80             | 7,079        | Entertainment     |1|
| App 4         | net.kreci.crackedscreen          | 3.20             | 7,547        | Entertainment     |1|
| App 5         | com.chartcross.gpstest           | 3.25             | 10,402       | Tools             |1|
| App 6         | com.unicornrobot.instantzing     | 3.33             | 26,882       | Game              |2|
| App 7         | com.laan.hacky                   | 4.00             | 2,708        | Game              |2|
| App 8         | com.kineticfoundry.ripple0beta   | 4.40             | 6,508        | Game              |2|
| App 9         | com.diotasoft.spark              | 4.50             | 2,345        | Game              |2|
| App 10        | com.m3roving.bettercrackedscreen | 4.80             | 8,065        | Entertainment     |2|
| App 11        | apps.powdercode.sailboat         | 5.18             | 28,662       | Game              |3|
| App 12        | cc.primevision.andosc            | 5.30             | 2,233        | Communication     |3|
| App 13        | com.distinctdev.tmtlite          | 7.33             | 14,641       | Game              |3|
| App 14        | com.hanoi                        | 7.67             | 4,598        | Game              |3|
| App 15        | com.groggy.cleanjokes.free       | 7.80             | 28,083       | Comics            |3|
| App 16        | bg.apps.randomstuff              | 8.75             | 8,451        | Tools             |4|
| App 17        | net.mandaria.tippytipper         | 8.83             | 4,858        | Finance           |4|
| App 18        | com.ts.sticks                    | 8.86             | 44,980       | Game              |4|
| App 19        | com.a4droid.sql\_reference       | 10.25            | 281,434      | Books & Reference |4|
| App 20        | com.twobitinc.cornholescore      | 12.71            | 4,421        | Entertainment     |4|
| App 21        | Gecko.Droid.PhysicsHelper        | 14.00            | 327,700      | Education         |5|
| App 22        | com.piviandco.fatbooth           | 17.36            | 16,658       | Entertainment     |5|
| App 23        | hr.podlanica                     | 23.30            | 23,549       | Music & Audio     |5|
| App 24        | com.roidapp.photogrid            | 23.30            | 12,558       | Photography       |5|
| App 25        | CapitalizationCalculator         | 23.60            | 9121         | Finance           |5|
| App 26        | com.tof.myquran                  | 25.54            | 41,349       | Books & Reference |6|
| App 27        | mobi.infolife.installer          | 30.91            | 52,975       | Business          |6|
| App 28        | com.cg.stickynote                | 31.00            | 341,608      | Productivity      |6|
| App 29        | com.pilot51.coinflip             | 31.89            | 16,700       | Game              |6|
| App 30        | com.youthhr.phonto               | 37.40            | 370,454      | Photography       |6|
| App 31        | conjugate.french.free            | 37.60            | 10,520       | Education         |7|
| App 32        | com.nixon.eval                   | 46.00            | 5,226        | Tools             |7|
| App 33        | nz.gen.geek\_central.ti5x        | 52.00            | 11,968       | Productivity      |7|
| App 34        | com.phellax.drum                 | 84.00            | 16,182       | Music & Audio     |7|
| App 35        | com.pandapow.vpn                 | 155.57           | 11,363       | Communication     |7|
| App 36        | com.naman14.stools               | 1.50             | 131,111      | Tools             |F-Droid|
| App 37        | com.secuso.torchlight2           | 2.50             | 127,752      | Tools             |F-Droid|
| App 38        | za.co.lukestonehm.logicaldefence | 3.00             | 192,093      | Education         |F-Droid|
| App 39        | de.ub0r.android.smsdroid         | 8.50             | 13,762       | Communication     |F-Droid|
| App 40        | com.bleyl.recurrence             | 35.75            | 249,326      | Productivity      |F-Droid|
| **Total**       | --                               | --        | 2,666,807    | --                | --|

## Acknowledgement

[GATOR](http://web.cse.ohio-state.edu/presto/software/gator/)
[Paladin](https://github.com/pkuoslab/Paladin)

