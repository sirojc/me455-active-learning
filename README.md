# ME455_ActiveLearning

*ME455 Active Learning in Robotics* repository of Joris Chomarat.

Spring 2023 @ Northwestern University

This class covered the fundamentals of automatically determining actions for obtaining information in robotic systems. Topics included methods in optimal control, probability and filtering theory, information theory and information measures, active sensing, infotaxis, nonparametric modeling techniques, and Markov Decision Processes. The following repository contains all assignments of this class, which apply most of the topics just named.

### [HW1](https://github.com/sirojc/me455-active-learning/blob/main/HW1/Homework1.pdf)

#### Finite Dimensional Numerical Optimization

Finite-dimensional numerical optimization applied to a differential drive vehicle.

<img src="https://github.com/sirojc/me455-active-learning/blob/main/HW1/Problem%201/HW1_1_init_traj.png" height="300"> <img src="https://github.com/sirojc/me455-active-learning/blob/main/HW1/Problem%201/HW1_1_opt_time.png" height="300">

#### Two Point Boundary Value Problem

TPBVP used to compute the control *u(t)* that minimizes a cost function subject to a dynamic constraint.

<img src="https://github.com/sirojc/me455-active-learning/blob/main/HW1/Problem%202/HW1_2_opt_time.png" height="300"> <img src="https://github.com/sirojc/me455-active-learning/blob/main/HW1/Problem%202/HW1_2_table.png" height="300">

#### Riccati Equation

Control *u(t)* solved for same problem as in TPBVP by solving the Riccati Equation. The plot displays the difference between the signals.

<img src="https://github.com/sirojc/me455-active-learning/blob/main/HW1/Problem%203/HW1_3_diff.png" height="300">

### [HW2](https://github.com/sirojc/me455-active-learning/blob/main/HW2/Homework2.pdf)

#### iLQR

An **i**terative **L**inear **Q**uadratic **R**egulator was applied to the differential drive vehicle from HW1, using the same semi-circle as initial trajectory.

<img src="https://github.com/sirojc/me455-active-learning/blob/main/HW2/HW2_1_traj.png" height="300"> <img src="https://github.com/sirojc/me455-active-learning/blob/main/HW2/HW2_1_opt_zoom.png" height="300">

### [HW3](https://github.com/sirojc/me455-active-learning/blob/main/HW3/Homework3.pdf)

#### Particle Filter

A particle filter was implemented for the same vehicle model as in HW1, using normally distirbuted process and measurement noise. The plot shows the estimated location for every second, the underlying *dt* being *0.1s*.

<img src="https://github.com/sirojc/me455-active-learning/blob/main/HW3/Problem%201/HW3_1_ext_est.png" height="300">

#### Kalman Filter

A Kalman filter was implemented in the following exercise.

<img src="https://github.com/sirojc/me455-active-learning/blob/main/HW3/Problem%202/HW3_Kalman_2_xvar.png" height="300">

#### Kalman vs. Nearby Filters

Using the Kalman filter from the prior exercise, it was numerically demonstrated that the Kalman filter is indeed the optimal linear filter, by comparing it to ten "nearby" filters for 100 smaple paths and looking at the average error for each filter.

<img src="https://github.com/sirojc/me455-active-learning/blob/main/HW3/Problem%203/HW3_3_e_std.png" height="300">

### [HW4](https://github.com/sirojc/me455-active-learning/blob/main/HW4/Homework4.pdf)

#### Ergodicity

The ergodic metric of a dynamic system with respect to the normally distributed distribution was computed, the dynamic system being dependent on a variable *b*.

<img src="https://github.com/sirojc/me455-active-learning/blob/main/HW4/Problem%201/HW4_1_ergodic_metric_bT.png" height="400">

#### Infotaxis Door Search

Infotaxis was implemented for the localization problem of a door, the likelihood function of the door being of hourglass shape.

<img src="https://github.com/sirojc/me455-active-learning/blob/main/HW4/Problem%202/plots/trajectory1.png" height="300"> <img src="https://github.com/sirojc/me455-active-learning/blob/main/HW4/Problem%202/plots/belief_run1.gif" width="430">

### [HW5](https://github.com/sirojc/me455-active-learning/blob/main/HW5/Homework5.pdf)

#### Ergodic Exploration

A maximally ergodic trajectory with respect to the normal distribution was computed for a dynamic system.

<img src="https://github.com/sirojc/me455-active-learning/blob/main/HW5/HW5_1_ergodic_traj.png" width="400"> <img src="https://github.com/sirojc/me455-active-learning/blob/main/HW5/HW5_1_cost.png" width="400">
