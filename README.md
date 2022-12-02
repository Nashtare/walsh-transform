# walsh-transform

This repository contains a PoC implementation of an exhaustive search of the maximum Fourier transform applied to the Anemoi S-box (Flystel) over bidimensional
vectors of a prime field Fp, in order to support a conjecture expressed in the Anemoi/Jive paper:
[New Design Techniques for Efficient Arithmetization-Oriented Hash Functions: Anemoi Permutations and Jive Compression Mode](https://eprint.iacr.org/2022/840.pdf).

We refer to this paper (see conjecture 1, Appendix 1) for more details on the Fourier transform and he Flystel S-Box of the Anemoi permutation.

This code is an adaptation of a sequential SAGEMATH implementation originally created by [Clemence Bouvier](https://who.rocq.inria.fr/Clemence.Bouvier/). It has been
adapted to Rust, optimized, and made parallel using `rayon`.

# LICENSE

This repository is [MIT licensed](LICENSE).