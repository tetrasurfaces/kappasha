.. This file is included into docs/history.rst

Hashlet

Hashlet is a fork of greenlet, extending lightweight coroutines with SHA256 hashing and RGB color mapping for Machine Environment Interface (MEI) visuals. It supports advanced control flow for concurrent programming, optimized for blockchain applications like arbitrage profitability checks and keyspace traversal using left-weighted sequences.

Features

Concurrent Hashing: Extends greenlet with SHA256 on greenlet ID/timestamp for immutable state tracking.

RGB Interoperations: Maps hashes to 24-bit RGB colors for MEI visualizations (e.g., chrysanthemum ramps).

Left-Weighted Sequence: Generates sequences (e.g., 11, 21, ..., 918) for efficient keyspace exploration, inspired by 16:7 (1:1) scale with 18 integers.

Profitability Checks: Rust-based and Solidity-based Δp * s > f calculations for on-chain arbitrage, with M53 Mersenne prime integration.

Applications: Ties to Greenpaper models (e.g., ternary ECC, Gaussian collapses) for blockchain and AI research.


Installation

pip install greenlet

Clone the repo:

git clone https://github.com/coneing/hashlet.git
cd hashlet

For Rust components:

cargo build --release

Usage



hashlet.py: Run concurrent tasks with hashing and RGB output.

python hashlet.py

Outputs: Hashes and RGB colors (e.g., #abcdef).



left_weighted_scale.py: Generate left-weighted sequences and balance scale placements.

python left_weighted_scale.py

Outputs: Sequence (e.g., [11, 21, ..., 918]) and weighing placements for 1-18.



lib.rs: Rust library for profitability checks (build as a crate).

cargo test

Outputs: Passes profitability tests for given inputs.


green.sol: Collapse Profitable M53


License

This repository is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). The lib.rs file is licensed under the MIT License for broader compatibility. The green.sol file is licensed under the MIT License for broader compatibility.

Contributing

Fork the repo, make changes, and submit pull requests. Focus areas:

Optimize MEI RGB mappings for finer entropy (~262K colors).

Extend left-weighted sequences for larger keyspaces.



What is a Greenlet?
===================

Greenlets are lightweight coroutines for in-process concurrent
programming.

The "greenlet" package is a spin-off of `Stackless`_, a version of
CPython that supports micro-threads called "tasklets". Tasklets run
pseudo-concurrently (typically in a single or a few OS-level threads)
and are synchronized with data exchanges on "channels".

A "greenlet", on the other hand, is a still more primitive notion of
micro-thread with no implicit scheduling; coroutines, in other words.
This is useful when you want to control exactly when your code runs.
You can build custom scheduled micro-threads on top of greenlet;
however, it seems that greenlets are useful on their own as a way to
make advanced control flow structures. For example, we can recreate
generators; the difference with Python's own generators is that our
generators can call nested functions and the nested functions can
yield values too. (Additionally, you don't need a "yield" keyword. See
the example in `test_generator.py
<https://github.com/python-greenlet/greenlet/blob/adca19bf1f287b3395896a8f41f3f4fd1797fdc7/src/greenlet/tests/test_generator.py#L1>`_).

Greenlets are provided as a C extension module for the regular unmodified
interpreter.

.. _`Stackless`: http://www.stackless.com


Who is using Greenlet?
======================

There are several libraries that use Greenlet as a more flexible
alternative to Python's built in coroutine support:

 - `Concurrence`_
 - `Eventlet`_
 - `Gevent`_

.. _Concurrence: http://opensource.hyves.org/concurrence/
.. _Eventlet: http://eventlet.net/
.. _Gevent: http://www.gevent.org/

Getting Greenlet
================

The easiest way to get Greenlet is to install it with pip::

  pip install greenlet


Source code archives and binary distributions are available on the
python package index at https://pypi.org/project/greenlet

The source code repository is hosted on github:
https://github.com/python-greenlet/greenlet

Documentation is available on readthedocs.org:
https://greenlet.readthedocs.io
