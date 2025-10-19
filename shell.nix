# shell.nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.rustc
    pkgs.cargo
    pkgs.gcc13
    pkgs.cudaPackages.cudatoolkit
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib64:$LD_LIBRARY_PATH
    echo "You are now in a reproducible Rust + CUDA + GCC13 shell!"
  '';
}
