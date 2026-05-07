# Implementation of Differentiable Timing-Driven Global Placement

This repository is my personal implementation attempt of the paper:

> Differentiable Timing-Driven Global Placement

The implementation is based on the open-source framework DREAMPlace 4.0.

I implemented differentiable timing propagation and related CUDA kernels mainly inside the `diff_timing` directory.  
Some CUDA kernels and implementation details were written with the help of AI assistance.

---

## Overview

The goal of this project was to reproduce the timing-driven optimization flow proposed in the paper using a differentiable timing formulation integrated into DREAMPlace.

For timing analysis:

- OpenTimer's dump graph functionality was used
- Timing arcs were extracted from the dumped graph
- RC tree propagation and differentiable timing propagation were implemented on top of the extracted graph

---

## Current Status

At the moment, the implementation runs, but timing optimization performance does not improve as expected.

Possible reasons include:

- incorrect hyperparameter settings
- implementation bugs
- incorrect gradient propagation
- scaling / unit conversion issues
- flaws in my interpretation of the paper

I checked:

- timing propagation logic
- unit conversions
- implementation flow described in the paper

but I still could not identify the main issue.

---

## Possible Typo in the Paper

While implementing the equations from the paper, I found what appears to be a typo in the gradient of delay equation.

Specifically:

- the original paper uses a `+` sign
- I implemented it using a `-` sign instead

because the derivative expansion seemed inconsistent otherwise.

This part may require further verification.

---

## Request for Feedback

I would greatly appreciate any feedback regarding:

- hyperparameter tuning
- WNS/TNS weighting strategy
- timing loss formulation
- gradient correctness
- timing propagation logic
- CUDA implementation issues
- overall optimization flow

There are many parts of the implementation that were assisted by AI-generated code, so there may be hidden implementation mistakes that I have not yet identified.

If you find incorrect logic or suspicious implementation details, please let me know.

---

## Notes

- Benchmark files are not included in this repository.
- This repository is intended mainly for experimentation and learning purposes.
- The implementation is still incomplete and under debugging.

---

## Acknowledgements

- DREAMPlace 4.0
- OpenTimer
- The authors of the original paper
