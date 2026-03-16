# Make Targets

This document standardizes command entry points.

## Prefix Convention

- `run-*`: start servers or execute full experiment chains.
- `bench-*`: run benchmark modes.
- `parse-*`: parse raw outputs into normalized tables.
- `dashboard-*`: build decision dashboard artifacts.
- `clean-*`: remove generated artifacts.

## Core Targets

- `make help`: list available targets.
- `make run-server-no-sd`: launch server without SD.
- `make run-server-eagle3`: launch server with EAGLE-3.
- `make run-baseline-no-sd`: run no-SD baseline chain.
- `make run-baseline-eagle3`: run EAGLE-3 baseline chain.
- `make run-bench CONFIG=<path>`: run benchmark from a config file.
- `make parse-results`: parse raw benchmark JSON files.
- `make dashboard-build`: build dashboard html and regression table.
- `make dashboard-readme`: refresh dashboard snapshot in README.
- `make clean-results`: clean generated result artifacts.
