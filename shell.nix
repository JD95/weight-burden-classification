with import <nixpkgs> {};

python36.withPackages (ps: [ps.Keras ps.matplotlib])