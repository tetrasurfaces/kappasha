# BlockChan Greenpaper Version Control

## Overview
The BlockChan Greenpaper uses a version control header at the top of `src/main.py` to track changes and ensure integrity. The `SHA-256 Hash` in the header is calculated based on the entire content of `src/main.py` and must be updated whenever the file is modified.

## Version Control Header
The header includes:
- **Artefact ID**: A unique identifier (`f2a3b4c5-d6e7-4f89-abcd-0123456789ef`).
- **SHA-256 Hash**: A hash of `src/main.py` (e.g., `bdf0cb3fed0f88be38014db4a9962a9dcc9553c31dba3f0d52d9cc95ace73cd5` as of September 21, 2025).
- **Date**: The date of the last update (e.g., September 21, 2025).
- **TOC Reference**: Reference to the Table of Contents (e.g., "0" - Whitepaper Overview).
- **Notes**: Detailed changelog and integration notes.
- **License**: GNU Affero General Public License v3.0 or later.
- **Publisher**: Anonymous.

## Hash Validation Process
1. Modify `src/main.py` as needed.
2. Run `python scripts/greenpaper_hasher.py` to calculate the new SHA-256 hash.
   - Example output: `Updated SHA-256 Hash: bdf0cb3fed0f88be38014db4a9962a9dcc9553c31dba3f0d52d9cc95ace73cd5`
3. Update the `SHA-256 Hash` field in the header with the new value.
4. Commit changes to version control with an appropriate message (e.g., "Updated hash to bdf0cb3... after TOC 41 demo integration").

## Additional Hashes
The project also tracks hashes of external content files (e.g., `green.txt`, `hybrid_cy.py`) defined in `src/config.py`. These are validated during runtime using the `validate_hashes` method in `GreenpaperUX`.

## Current Version
- **Version**: 2.9
- **Date**: September 21, 2025
- **Hash**: bdf0cb3fed0f88be38014db4a9962a9dcc9553c31dba3f0d52d9cc95ace73cd5
