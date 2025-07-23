# EntropicLearning.jl

[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

An MLJ-compatibile implementation of various entropic learning methods. It currently includes:

- eSPA+, described in [[1]](#1) and [[2]](#2)

This implementation is based on the Python implementation by Davide Bassetti (https://github.com/davbass/entlearn), as well as the original MATLAB implementation by Illia Horenko (https://github.com/horenkoi/eSPA)

## License

This software is distributed under the Academic Software License (ASL), which is included in the LICENSE file. Please familiarise yourself with this license before using the code.

## Citing

If you find this package useful, please cite the original publications describing the methods ([[1]](#1), [[2]](#2)) and this repository. Thanks! 

## References

<a id="1">[1]</a> Horenko, I. (2020). On a scalable entropic breaching of the overfitting barrier for small data problems in machine learning. Neural Computation, 32(8), 1563-1579.

<a id="2">[2]</a> Vecchi, E., Pospíšil, L., Albrecht, S., O'Kane, T. J., & Horenko, I. (2022). eSPA+: Scalable entropy-optimal machine learning classification for small data problems. Neural Computation, 34(5), 1220-1255.
