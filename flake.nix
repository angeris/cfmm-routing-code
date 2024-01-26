{
  description = "CFFM routing";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    scip = {
      url = github:dzmitry-lahoda-forks/scip/d3d14840f8b87afbf9e6a7064cb7dbae7e386622;
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyscipopt-src = {
      url = "github:scipopt/PySCIPOpt/v4.4.0";
      flake = false;
    };
    maturin-src = {
      url = "github:PyO3/maturin";
      flake = false;
    };
  };

  outputs = inputs@{ flake-parts, poetry2nix, pyscipopt-src, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ];
      perSystem = { config, self', inputs', pkgs, system, ... }:
        let
          maturin-latest = pkgs.python3Packages.buildPythonPackage {
            name = "maturin";
            version = "0.0.1";
            format = "pyproject";

            src = inputs.maturin-src;

            cargoDeps = pkgs.rustPlatform.fetchCargoTarball {
              src = inputs.maturin-src;
              name = "maturin";
              version = "0.0.1";
              hash = "sha256-3zPG/6EQiXaDaxgNYIAsJ5A7MbbK2Gxj8NDpEukUit1=";
            };

            nativeBuildInputs = with pkgs.python3Packages; [
              poetry-core
              setuptools
              setuptools-rust
              setuptools-git-versioning
              pkgs.rustPlatform.cargoSetupHook
              pkgs.rustPlatform.maturinBuildHook
            ];
          };
          scipopt = [
            inputs'.scip.packages.scip
            inputs'.scip.packages.soplex
            inputs'.scip.packages.papilo
          ];
          nativeBuildInputs = with pkgs; [
            poetry
            python3
            zlib
            zlib.dev
            pkg-config
            (texliveSmall.withPackages
              (ps: with ps; [ gensymb type1cm cm-super ]))
            ps
            pyscipopt
            envShell
          ] ++ scipopt;
          inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryEnv overrides;
          override = overrides:
            overrides.withDefaults (self: super: {
              clarabel = super.clarabel.overridePythonAttrs (old: {
                buildInputs = old.buildInputs or [ ] ++ [ self.python.pkgs.maturin ];
              });
              scs = super.scs.overridePythonAttrs (old: {
                buildInputs = old.buildInputs or [ ] ++ [ pkgs.meson pkgs.python3Packages.meson-python pkgs.pkg-config pkgs.blas pkgs.lapack ];
                nativeBuildInputs = old.nativeBuildInputs or [ ] ++ [ pkgs.meson pkgs.python3Packages.meson-python pkgs.blas pkgs.lapack ];
              });
              pyscipopt = pyscipopt;
              # cylp = super.cylp.overridePythonAttrs (old: {
              #   buildInputs = old.buildInputs or [] ++ [self.python.pkgs.setuptools self.python.pkgs.wheel pkgs.cbc pkgs.pkg-config];
              #   nativeBuildInputs = old.nativeBuildInputs or [] ++ [self.python.pkgs.setuptools self.python.pkgs.wheel pkgs.cbc pkgs.pkg-config];
              # });
              # cvxpy = cvxpy-latest;
              maturin = maturin-latest;
            });

          envShell = mkPoetryEnv {
            projectDir = ./.;
            overrides = override overrides;
            extraPackages = (ps: [ pyscipopt ]);
          };
          pyscipopt = pkgs.python3Packages.buildPythonPackage {
            name = "pyscipopt";
            version = "v4.4.0";
            format = "pyproject";
            LD_LIBRARY_PATH = with pkgs; lib.strings.makeLibraryPath [
              "${inputs'.scip.packages.scip}/lib"
            ];
            SCIPOPTDIR = inputs'.scip.packages.scip;
            src = inputs.pyscipopt-src;
            propagatedBuildInputs = scipopt;
            nativeBuildInputs = with pkgs.python3Packages; [
              setuptools
              pkgs.pkg-config
              pkgs.python311Packages.cython
            ] ++ scipopt;
            buildInputs = with pkgs.python3Packages; [
              cython
            ] ++ scipopt;
          };
        in
        {
          packages = {
            inherit pyscipopt;
            scip = inputs'.scip.packages.scip;
            soplex = inputs'.scip.packages.soplex;
            papilo = inputs'.scip.packages.papilo;
          };
          devShells.default = pkgs.mkShell {
            LD_LIBRARY_PATH = with pkgs; lib.strings.makeLibraryPath [
              stdenv.cc.cc.lib
              zlib
              zlib.dev
              zlib.out
              #"${inputs'.scip.packages.scip}/lib"
            ];
            #SCIPOPTDIR = "asdsada";# inputs'.scip.packages.scip;
            inherit nativeBuildInputs;

            enterShell = ''
              poetry lock --no-update
              poetry instal --no-root
              poetry run black numerics-two-assets.py
            '';
          };
        };
    };
}
